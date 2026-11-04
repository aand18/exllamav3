"""
RowTable's decode-size row cache (EXL3_NGRAM_ROW_CACHE_ROWS): a gather of a few rows keeps the rows
it read, keyed by the row's global index, so a repeat does not become another unbuffered NVMe read
on the decode critical path. What these tests pin down:

- a row served from the cache is byte-identical to the same row read off the disk (the uncached
  gather is the oracle, and the reference read in fetch() a second one),
- the cache never holds more rows than its budget, however many distinct rows go through it,
- a gather larger than the decode threshold (a prefill chunk) neither looks a row up nor evicts the
  cache,
- with the knob unset the table allocates no cache at all and counts nothing.

A tiny row table file is the whole fixture: no model, no device, no safetensors header.
"""

import pytest
import torch

from exllamav3.loader.safetensors import DiskTensorHandle
from exllamav3.modules import row_table as rt

pytestmark = pytest.mark.nogpu

ROWS = 128          # rows in the test table, more than the decode threshold
WORDS = 64          # int16 words per row: 128 B, the scale of a quantized n-gram row
BASE = 512          # the rows start off a sector boundary, as they do inside a safetensors file
FILE_BYTES = 20480  # a multiple of the 4 KiB unit an unbuffered Windows read aligns to
BUDGET = 4


@pytest.fixture
def table(tmp_path):
    """A table of ROWS random int16 rows in a file, streamed through a DiskTensorHandle. Built with
    _set_stores(), as open() would leave it: the gather path under test starts below that."""
    g = torch.Generator().manual_seed(0)
    data = torch.randint(-32768, 32768, (ROWS, WORDS), generator = g, dtype = torch.int16)
    path = tmp_path / "ngram.bin"
    path.write_bytes(bytes(BASE) + data.numpy().tobytes() + bytes(FILE_BYTES - BASE - data.nbytes))
    t = rt.RowTable("test.ngram", ["test.ngram.weight"])
    t._set_stores([DiskTensorHandle(key = "test.ngram.weight", filename = str(path),
                                    abs_offset = BASE, shape = [ROWS, WORDS], dtype = torch.int16)])
    yield t
    t.close()


def gather(table, rows):
    """One _gather_rows over these global row ids, in the shape lookup() hands it: sorted unique uids
    and a pinned staging buffer of the table's row shape. Returns the staged rows, in uid order."""
    uids = torch.tensor(rows, dtype = torch.int64)
    out = torch.empty((len(rows), table.row_words), dtype = table.row_dtype, pin_memory = True)
    table._gather_rows(uids, out)
    return out


def test_a_cached_row_is_byte_identical(table, monkeypatch):
    monkeypatch.delenv("EXL3_NGRAM_ROW_CACHE_ROWS", raising = False)
    rt.reset_row_cache()
    uncached = gather(table, [7])
    assert rt.row_cache_hits() == 0 and rt.row_cache_misses() == 0
    assert torch.equal(uncached, table.fetch(torch.tensor([7])))   # the reference read agrees

    monkeypatch.setenv("EXL3_NGRAM_ROW_CACHE_ROWS", str(BUDGET))
    rt.reset_row_cache()
    first = gather(table, [7])
    assert rt.row_cache_misses() == 1 and rt.row_cache_hits() == 0
    second = gather(table, [7])
    assert rt.row_cache_hits() == 1 and rt.row_cache_misses() == 1
    assert torch.equal(first, uncached)
    assert torch.equal(second, uncached)


def test_cache_never_exceeds_its_row_budget(table, monkeypatch):
    monkeypatch.setenv("EXL3_NGRAM_ROW_CACHE_ROWS", str(BUDGET))
    rt.reset_row_cache()
    gather(table, list(range(16)))
    assert rt.row_cache_misses() == 16
    assert len(table._row_cache) <= BUDGET
    # the rows that survived are the ones gathered last: the recent row is served, not re-read
    gather(table, [15])
    assert (rt.row_cache_hits(), rt.row_cache_misses()) == (1, 16)
    # a mixed gather: rows 0 and 1 were evicted (read again, to the same bytes the reference read
    # gives) while 15 is a hit, so the run of reads and the copies interleave in uid order
    mixed = gather(table, [0, 1, 15])
    assert torch.equal(mixed, table.fetch(torch.tensor([0, 1, 15])))
    assert (rt.row_cache_hits(), rt.row_cache_misses()) == (2, 18)
    assert len(table._row_cache) <= BUDGET


def test_a_bulk_lookup_bypasses_the_cache(table, monkeypatch):
    monkeypatch.setenv("EXL3_NGRAM_ROW_CACHE_ROWS", str(BUDGET))
    rt.reset_row_cache()
    bulk = gather(table, list(range(rt.ROW_CACHE_GATHER_MAX_ROWS + 1)))
    assert rt.row_cache_hits() == 0 and rt.row_cache_misses() == 0
    assert not hasattr(table, "_row_cache")
    assert torch.equal(bulk, table.fetch(torch.arange(rt.ROW_CACHE_GATHER_MAX_ROWS + 1)))


def test_the_cache_is_off_by_default(table, monkeypatch):
    monkeypatch.delenv("EXL3_NGRAM_ROW_CACHE_ROWS", raising = False)
    rt.reset_row_cache()
    gather(table, [3])
    gather(table, [3])      # a repeated row would be a hit if any cache were running
    assert rt.row_cache_hits() == 0 and rt.row_cache_misses() == 0
    assert not hasattr(table, "_row_cache")
