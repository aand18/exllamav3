"""
Pinned hot set on top of RowTable's row cache (EXL3_NGRAM_ROW_CACHE_PIN): a file of row ids that the
cache loads at open and never evicts. Where test_row_cache.py covers the clock (a row survives one
lap of the hand if it was looked up), this covers the rows that have to survive every lap.

The claim the pin exists for: the clock is a recency policy, so a row the whole corpus shares with
the tail of a 20M-row table is exactly the row that competes with it for four slots and loses. What
these tests pin down:

- a pinned row is a hit after two full laps of the LRU, which is where an unpinned copy of it dies,
- a pin set that does not fit the row budget is a startup error naming the count and the budget,
- so is a pin set over EXL3_NGRAM_ROW_CACHE_PIN_MAX_BYTES (the L3 guard), naming both byte counts,
- a pinned lookup is counted as a hit, and row_cache_pinned() reports how many rows are pinned,
- with the knobs unset nothing is allocated and nothing is pinned, byte for byte today's behaviour.

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
ROW_BYTES = WORDS * 2


@pytest.fixture
def table(tmp_path):
    """A table of ROWS random int16 rows in a file, streamed through a DiskTensorHandle. Built with
    _set_stores(), as open() would leave it: the pin is loaded by _pin_row_cache(), which is what
    open() calls once the stores exist (and which a gather reaches through _gather_row_cache())."""
    g = torch.Generator().manual_seed(0)
    data = torch.randint(-32768, 32768, (ROWS, WORDS), generator = g, dtype = torch.int16)
    path = tmp_path / "ngram.bin"
    path.write_bytes(bytes(BASE) + data.numpy().tobytes() + bytes(FILE_BYTES - BASE - data.nbytes))
    t = rt.RowTable("test.ngram", ["test.ngram.weight"])
    t._set_stores([DiskTensorHandle(key = "test.ngram.weight", filename = str(path),
                                    abs_offset = BASE, shape = [ROWS, WORDS], dtype = torch.int16)])
    yield t
    t.close()


def pin_file(tmp_path, rows, name = "pin.txt"):
    """A pin file in the format EXL3_NGRAM_ROW_CACHE_PIN reads, comments and blanks included: the
    offline tool writes the same shape, and the engine must not need a second parser for it."""
    path = tmp_path / name
    body = "# hot rows, one per line\n" + "".join(f"{r}\n" for r in rows) + "\n# end\n"
    path.write_text(body, encoding = "utf-8")
    return path


def gather(table, rows):
    """One _gather_rows over these global row ids, as lookup() hands it to the gather."""
    uids = torch.tensor(rows, dtype = torch.int64)
    out = torch.empty((len(rows), table.row_words), dtype = table.row_dtype, pin_memory = True)
    table._gather_rows(uids, out)
    return out


def enable(monkeypatch, tmp_path, pins, budget = BUDGET, max_bytes = None):
    monkeypatch.setenv("EXL3_NGRAM_ROW_CACHE_ROWS", str(budget))
    if pins is None:
        monkeypatch.delenv("EXL3_NGRAM_ROW_CACHE_PIN", raising = False)
    else:
        monkeypatch.setenv("EXL3_NGRAM_ROW_CACHE_PIN", str(pin_file(tmp_path, pins)))
    if max_bytes is None:
        monkeypatch.delenv("EXL3_NGRAM_ROW_CACHE_PIN_MAX_BYTES", raising = False)
    else:
        monkeypatch.setenv("EXL3_NGRAM_ROW_CACHE_PIN_MAX_BYTES", str(max_bytes))


def test_a_pinned_row_is_never_evicted(table, tmp_path, monkeypatch):
    """One pin, three unpinned slots, six distinct rows: two full laps of the hand. The pinned row is
    looked up only after the laps, so it cannot be surviving on recency - with the hand's second
    chance it is the only row left to evict on the second lap, and an unpinned copy of it goes."""
    enable(monkeypatch, tmp_path, [3])
    rt.reset_row_cache()
    table._pin_row_cache()
    assert rt.row_cache_pinned() == 1

    for row in range(10, 16):
        gather(table, [row])
    assert (rt.row_cache_hits(), rt.row_cache_misses()) == (0, 6)

    pinned = gather(table, [3])
    assert rt.row_cache_hits() == 1          # a hit: no read, no miss
    assert rt.row_cache_misses() == 6
    assert rt.row_cache_pinned() == 1
    assert torch.equal(pinned, table.fetch(torch.tensor([3])))


def test_a_pin_file_larger_than_the_budget_is_an_error(table, tmp_path, monkeypatch):
    """Not a silent drop of the rows that do not fit: a pin set the budget cannot hold is a
    misconfiguration, and the error names both numbers so the fix is in the message."""
    enable(monkeypatch, tmp_path, list(range(BUDGET + 1)))
    with pytest.raises(ValueError) as err:
        table._pin_row_cache()
    msg = str(err.value)
    assert "EXL3_NGRAM_ROW_CACHE_PIN" in msg and "EXL3_NGRAM_ROW_CACHE_ROWS" in msg
    assert f"{BUDGET + 1} rows" in msg and f"={BUDGET}" in msg
    assert not hasattr(table, "_row_cache")      # nothing half-loaded


def test_the_byte_guard_errors_by_name(table, tmp_path, monkeypatch):
    """The guard that keeps the pinned set inside the CPU's L3: a byte limit one below the set's own
    size refuses to start, naming the limit and the set's total."""
    pins = [1, 2]
    total = len(pins) * ROW_BYTES
    enable(monkeypatch, tmp_path, pins, max_bytes = total - 1)
    with pytest.raises(ValueError) as err:
        table._pin_row_cache()
    msg = str(err.value)
    assert "EXL3_NGRAM_ROW_CACHE_PIN_MAX_BYTES" in msg
    assert f"{total} B" in msg and f"{total - 1} B" in msg


def test_pinned_rows_count_as_hits(table, tmp_path, monkeypatch):
    """A pin is not a side table: the lookup path serves it, so it shows up in the hit count the
    measurement reads, and the pin count is reported alongside."""
    enable(monkeypatch, tmp_path, [5, 6])
    rt.reset_row_cache()
    table._pin_row_cache()
    assert rt.row_cache_pinned() == 2

    out = gather(table, [5])
    assert (rt.row_cache_hits(), rt.row_cache_misses()) == (1, 0)
    assert rt.row_cache_pinned() == 2            # a hit does not change the pin set
    assert torch.equal(out, table.fetch(torch.tensor([5])))
    assert "pinned=2" in table._row_cache.report()


def test_the_pin_is_off_by_default(table, monkeypatch):
    """No knobs: no cache object, nothing pinned, no counters - the behaviour of the table before the
    pin existed, which is what every existing deployment runs."""
    for knob in ("EXL3_NGRAM_ROW_CACHE_ROWS", "EXL3_NGRAM_ROW_CACHE_PIN",
                 "EXL3_NGRAM_ROW_CACHE_PIN_MAX_BYTES"):
        monkeypatch.delenv(knob, raising = False)
    rt.reset_row_cache()
    gather(table, [3])
    gather(table, [3])          # a repeat would be a hit if any cache were running
    assert not hasattr(table, "_row_cache")
    assert rt.row_cache_pinned() == 0
    assert (rt.row_cache_hits(), rt.row_cache_misses()) == (0, 0)
