"""
File handles held by a synthetic checkpoint the test helpers build.

make_stc / module_config hand out a real SafetensorsCollection, which keeps one open stream per file
and per loader thread (eight threads on Windows) for as long as it is alive. A module test builds one
per case, and the loader never closes a collection by itself, so on a platform with a ~512-entry
descriptor table the table ran out about sixty cases into tests/modules and every test after that one
failed with "Too many open files (errno=24)", whatever it was testing: 131 of 145 failures were that
one cascade. The helper owns the collection, so the helper releases it when the last reference dies
(testlib.checkpoint.SelfClosingCollection).

Two properties are pinned here. A collection nothing refers to any more must not hold the process's
handles, and a collection something still refers to must, because a module-scoped fixture's collection
is used by every later test in its module and closing it early breaks them.
"""

import gc
import os
from types import SimpleNamespace

import pytest
import torch

from testlib.checkpoint import make_stc

pytestmark = pytest.mark.nogpu

N_COLLECTIONS = 256
SLACK = 16


def open_handles() -> int:
    """Handles the process holds: kernel handles on Windows (one per open CRT stream), descriptors on Linux"""
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessHandleCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        count = ctypes.c_ulong()
        if not kernel32.GetProcessHandleCount(kernel32.GetCurrentProcess(), ctypes.byref(count)):
            pytest.skip("GetProcessHandleCount is unavailable on this host")
        return int(count.value)
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        pytest.skip("no /proc/self/fd to count descriptors on this platform")


def load_one(directory):
    stc = make_stc({"a": torch.zeros(4)}, directory)
    stc.get_tensor("a", torch.device("cpu"))
    return stc


def test_a_collection_nothing_refers_to_holds_no_handles(tmp_path):
    load_one(tmp_path / "warmup")           # first-call allocations should not count against the baseline
    gc.collect()
    baseline = open_handles()

    # The measurement is only meaningful if a live collection does hold its file open
    stc = load_one(tmp_path / "live")
    assert open_handles() > baseline
    del stc
    gc.collect()

    # One collection per case, the way a module test builds them: each is released as the name is rebound
    for i in range(N_COLLECTIONS):
        load_one(tmp_path / f"c{i}")
    gc.collect()
    assert open_handles() <= baseline + SLACK


def test_a_collection_still_in_use_keeps_its_file(tmp_path):
    # A module-scoped fixture's config, for the tests that load from it afterwards
    config = SimpleNamespace(stc = load_one(tmp_path))
    gc.collect()
    assert config.stc.get_tensor("a", torch.device("cpu")).tolist() == [0, 0, 0, 0]
