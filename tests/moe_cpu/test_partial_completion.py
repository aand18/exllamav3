"""
Per-expert partial completion (T57, EXL3_MOE_PARTIAL_COMPLETION) on the parent's side, without a GPU: the layout
arithmetic of the three mirrors (cpu/moe_handoff.h, moe_cpu_host.py and testlib/moe_handoff.py), the pinned-slot
budget the flag costs, and the collect sequence -- one wait/fold pair per reserved ring position, in submission order,
then exactly one consumed write for the slot.

The fold itself (moe_split_collect_add_expert) and the announce kernel are covered by tests/kernels/moe/test_moe_split.py
and tests/kernels/moe_cpu_host/test_moe_worker.py, which need a device and the compiled extension.
"""

import contextlib
import os
import re
from types import SimpleNamespace

import pytest

from exllamav3.model import moe_cpu_host as host

pytestmark = pytest.mark.nogpu

HDR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "exllamav3", "exllamav3_ext", "cpu", "moe_handoff.h")

# Layout constants that exist in every mirror (the C++ header is the source of truth)
LAYOUT_NAMES = ("MOE_JOB_RING", "MOE_MAX_SLOTS", "MOE_MAX_WSLOTS", "MOE_STAGE_RING", "MOE_CTRL_JOBS_OFFSET",
                "MOE_SLOT_FLAGS_OFFSET", "MOE_FLAGS_SIZE", "MOE_EXPERT_DONE_OFFSET", "MOE_STAGE_FLAGS_OFFSET",
                "MOE_STAGE_TAIL_OFFSET", "MOE_STAGE_HEAD_OFFSET", "MOE_STAGE_JOBS_OFFSET", "MOE_CTRL_SIZE")


def header_defines() -> dict:
    """Numeric #defines of the handoff header, evaluated: comments stripped, sizeof(MoeJob) replaced by the Python
    mirror's MOE_JOB_BYTES (the one struct size that is derived rather than written down in the header)"""
    text = open(HDR).read()
    text = re.sub(r"/\*.*?\*/", "", text, flags = re.S)
    text = re.sub(r"//[^\n]*", "", text)
    out = {}
    for name, expr in re.findall(r"#define\s+(MOE_[A-Z0-9_]+)[ \t]+([^\n]+)", text):
        expr = expr.strip().replace("sizeof(MoeJob)", f"({host.MOE_JOB_BYTES})")
        expr = expr.rstrip("uU")
        try:
            out[name] = int(eval(expr, {"__builtins__": {}}, dict(out)))
        except Exception:
            continue        # not a plain integer expression (conditional flag width, casts)
    return out


def slot_layout(partial: bool, cap_rows = 64, topk = 8, hi = 4096, ho = 4096, e_cpu = 256) -> dict:
    """ensure_started's slot arithmetic (moe_cpu_host.py) for one configuration"""
    align64 = host._align64
    off_x = 0
    off_sel = align64(off_x + cap_rows * hi * 2)
    off_w = align64(off_sel + cap_rows * topk * 4)
    off_out = align64(off_w + cap_rows * topk * 2)
    out_rows = cap_rows * topk if partial else cap_rows
    off_rowidx = align64(off_out + out_rows * ho * 4)
    off_segtab = align64(off_rowidx + out_rows * 4)
    slot_size = align64(off_segtab + 2 * e_cpu * 4) if partial else align64(off_out + out_rows * ho * 4)
    return dict(off_out = off_out, off_rowidx = off_rowidx, off_segtab = off_segtab, out_rows = out_rows,
                slot_size = slot_size)


def test_python_mirrors_match_the_cxx_header():
    hdr = header_defines()
    import testlib.moe_handoff as tlib
    for name in LAYOUT_NAMES:
        assert name in hdr, f"{name} is not a numeric #define in cpu/moe_handoff.h"
        assert getattr(host, name) == hdr[name], f"{name}: moe_cpu_host.py {getattr(host, name)} != header {hdr[name]}"
        assert getattr(tlib, name) == hdr[name], f"{name}: testlib {getattr(tlib, name)} != header {hdr[name]}"
    # Per-expert job fields and markers, mirrored by name in the parent and the test lib
    assert hdr["MOE_JOB_KIND_COMPUTE_EXPERT"] == host.MOE_JOB_KIND_COMPUTE_EXPERT == tlib.KIND_COMPUTE_EXPERT
    assert hdr["MOE_EXPERT_SKIP"] == host.MOE_EXPERT_SKIP == tlib.EXPERT_SKIP
    assert hdr["MOE_JOB_MAX_EXPERTS"] == host.MOE_JOB_MAX_EXPERTS


def test_flag_banks_are_contiguous_and_leave_room_for_the_ring():
    # data_ready, done, consumed (per slot), expert_done (per ring position), then the two stager banks
    assert host.MOE_EXPERT_DONE_OFFSET == host.MOE_SLOT_FLAGS_OFFSET + 3 * 64 * host.MOE_MAX_SLOTS
    assert host.MOE_STAGE_FLAGS_OFFSET == host.MOE_EXPERT_DONE_OFFSET + 64 * host.MOE_JOB_RING
    assert host.MOE_STAGE_TAIL_OFFSET == host.MOE_SLOT_FLAGS_OFFSET + host.MOE_FLAGS_SIZE
    assert host.MOE_STAGE_FLAGS_OFFSET + 2 * 64 * host.MOE_MAX_WSLOTS == host.MOE_STAGE_TAIL_OFFSET
    assert host.MOE_CTRL_SIZE == host.MOE_STAGE_JOBS_OFFSET + host.MOE_STAGE_RING * host.MOE_JOB_BYTES
    assert host.MOE_CTRL_SIZE < 3 << 20, "the control block is per host, but it is pinned: keep it ~2 MiB"


def test_partial_slot_layout_and_pinned_budget():
    """The per-expert path costs a slice row per (token, expert) assignment plus a small row idx and segment table;
    nothing of it is VRAM and none of it is paid when the flag is off"""
    whole = slot_layout(False)
    part = slot_layout(True)
    assert whole["out_rows"] == 64 and part["out_rows"] == 64 * 8
    # Slice region, token rows and (offset, count) table, each 64-byte aligned after the previous
    assert part["off_out"] == whole["off_out"], "the sections before out are shared by both layouts"
    assert part["off_rowidx"] == part["off_out"] + part["out_rows"] * 4096 * 4
    assert part["off_segtab"] == part["off_rowidx"] + part["out_rows"] * 4
    assert part["slot_size"] == (part["off_segtab"] + 2 * 256 * 4 + 63) & ~63
    # The delta is the extra slice rows (topk - 1 per token) plus the two small tables
    expected = (whole["off_out"] + 512 * 4096 * 4 + 512 * 4 + 2048 + 63) & ~63
    assert part["slot_size"] == expected
    assert part["slot_size"] < 9 << 20, "one slot stays under 9 MiB at cap 64 x topk 8"
    # A layer reserves cap_rows * topk ring positions at once and the throttle keeps 4 in hand
    assert 64 * 8 + 4 <= host.MOE_JOB_RING


@pytest.mark.parametrize("cap_rows, topk", [(8, 3), (64, 8), (32, 16), (127, 16)])
def test_ring_budget_holds_for_the_supported_shapes(cap_rows, topk):
    assert cap_rows * topk + 4 <= host.MOE_JOB_RING


def test_partial_completion_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv("EXL3_MOE_PARTIAL_COMPLETION", raising = False)
    assert host.MoeCpuTuning().partial_completion is False
    monkeypatch.setenv("EXL3_MOE_PARTIAL_COMPLETION", "1")
    assert host.MoeCpuTuning().partial_completion is True
    monkeypatch.setenv("EXL3_MOE_PARTIAL_COMPLETION", "0")
    assert host.MoeCpuTuning().partial_completion is False


class _ExtRecorder:
    """Stands in for the compiled extension: records the flag/fold calls the collect enqueues, in order"""

    def __init__(self):
        self.calls = []

    def exl3_moe_flag_wait(self, flag, value, abort):
        self.calls.append(("wait", flag, value))

    def moe_split_collect_add_expert(self, out, h_out, h_rowidx, h_segtab, h_expert, ho, n_cap):
        self.calls.append(("fold", out, h_out, h_rowidx, h_segtab, h_expert, ho, n_cap))

    def exl3_moe_flag_write(self, flag, value):
        self.calls.append(("write", flag, value))


def _stub_host(cap_rows = 4, topk = 3, ho = 128, e_cpu = 4, num_slots = 2) -> SimpleNamespace:
    """Just what _collect_partial touches: the slot views' device aliases and the segment base"""
    base = 0x100000
    sbase = 0x2000
    slots = [dict(out_dev = base + sbase + s * 0x1000, rowidx_dev = base + sbase + s * 0x1000 + 0x800,
                  segtab_dev = base + sbase + s * 0x1000 + 0xC00, consumed = base + 0x900 + 64 * s)
             for s in range(num_slots)]
    return SimpleNamespace(slots = slots, gpu_base_ptr = base, partial_cap = cap_rows * topk,
                           max_experts = e_cpu, num_slots = num_slots, seq = 0)


def _collect(monkeypatch, stub, handle):
    """Run MoeCpuHost._collect_partial against the recorder, with torch.cuda.device stubbed out (nogpu)"""
    rec = _ExtRecorder()
    monkeypatch.setattr(host, "ext", rec)
    monkeypatch.setattr(host.torch.cuda, "device", lambda dev: contextlib.nullcontext())
    final = object()
    stub._final = final
    host.MoeCpuHost._collect_partial(stub, handle, final)
    return rec.calls


def test_combine_enqueues_one_wait_and_fold_per_job_then_one_consumed(monkeypatch):
    cap_rows, topk = 4, 3
    stub = _stub_host(cap_rows, topk)
    seq, slot_idx, tail = 7, 1, 11
    cap = cap_rows * topk
    calls = _collect(monkeypatch, stub, ("partial", seq, slot_idx, cap_rows, 32, 64, "cuda:0", tail, cap))
    assert len(calls) == 2 * cap + 1
    base = stub.gpu_base_ptr
    for i in range(cap):
        pos = (tail + i) % host.MOE_JOB_RING
        wait, fold = calls[2 * i], calls[2 * i + 1]
        # The wait precedes its fold, both keyed by the ring position, so folds land in submission order
        # regardless of which expert finishes first
        assert wait == ("wait", base + host.MOE_EXPERT_DONE_OFFSET + pos * 64, seq)
        assert fold == ("fold", stub._final, stub.slots[slot_idx]["out_dev"], stub.slots[slot_idx]["rowidx_dev"],
                        stub.slots[slot_idx]["segtab_dev"],
                        base + host.MOE_CTRL_JOBS_OFFSET + pos * host.MOE_JOB_BYTES + 4 * host.MOE_JOB_EXPERTS0,
                        64, cap)
    # ONE consumed write, for the whole slot, after the last fold (slot reuse waits for every fold)
    assert calls[-1] == ("write", stub.slots[slot_idx]["consumed"], seq)


def test_combine_wraps_the_ring_and_folds_the_whole_reservation(monkeypatch):
    cap_rows, topk = 2, 2
    stub = _stub_host(cap_rows, topk)
    tail = host.MOE_JOB_RING - 2      # reservation straddles the ring end
    cap = cap_rows * topk
    calls = _collect(monkeypatch, stub, ("partial", 3, 0, cap_rows, 0, 64, "cuda:0", tail, cap))
    positions = [(tail + i) % host.MOE_JOB_RING for i in range(cap)]
    assert positions == [host.MOE_JOB_RING - 2, host.MOE_JOB_RING - 1, 0, 1]
    assert [c[1] for c in calls if c[0] == "wait"] == \
        [stub.gpu_base_ptr + host.MOE_EXPERT_DONE_OFFSET + p * 64 for p in positions]
    assert len([c for c in calls if c[0] == "write"]) == 1
