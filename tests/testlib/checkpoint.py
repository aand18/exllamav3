"""
Synthetic checkpoints served through the real loader.

Module tests build a module from a dict of tensors. Rather than a dict-backed stand-in for the safetensors
collection (which drifts from the real loader's signature and semantics), the tensors are written to a
safetensors file and read back through SafetensorsCollection, so codebook markers, arena allocation, padding
and dtype handling are the production code paths.

    stc = make_stc(tensors, tmp_path)
    config = module_config(tensors, tmp_path)       # has .stc and .infer_params, enough for Module(config, ...)

The collection closes its own file handles when the last reference to it dies (SelfClosingCollection),
so a test case can build one without the process's descriptor table filling up over a long run.
"""

import os
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from exllamav3.loader.safetensors import SafetensorsCollection


class SelfClosingCollection(SafetensorsCollection):
    """
    A collection that releases its file handles when the last reference to it dies.

    Nothing in the loader closes a collection by itself, and a module test builds one per case. Each
    open collection holds one open stream per loader thread (STLOADER_THREADS, eight on Windows), so
    on a platform with a ~512-entry descriptor table the table runs out after about sixty cases and
    every later test in the process fails with "Too many open files (errno=24)".

    The close cannot be scoped to the test that built the collection: a collection built by a
    module-scoped fixture is still in use by the tests that follow it in that module. It cannot be
    a weakref.finalize either, since passing the collection as the finalizer's argument keeps it
    alive and the callback never runs. __del__ is the one hook that sees the collection at the
    moment its last reference goes away.
    """

    def __del__(self):
        try:
            # A conversion's set_new_tensors() window owns the collection's tensors, and close()
            # asserts on that state; the code that opened the window closes it
            if self.__dict__.get("new_tensors") is None:
                self.close()
        except Exception:
            # __del__ must not raise: it also runs at interpreter shutdown, after the loader's
            # module globals and the extension may already have been torn down
            pass


def write_tensors(tensors: dict[str, torch.Tensor], directory: str, filename: str = "model.safetensors") -> str:
    os.makedirs(directory, exist_ok = True)
    path = os.path.join(directory, filename)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, path)
    return path


def make_stc(tensors: dict[str, torch.Tensor], directory) -> "SafetensorsCollection":
    directory = str(directory)
    write_tensors(tensors, directory)
    return SelfClosingCollection(directory)


def module_config(tensors: dict[str, torch.Tensor], directory, **infer_params):
    """Minimal config for constructing and loading standalone modules: .stc and .infer_params"""
    from exllamav3.model.config import InferParams
    ip = InferParams()
    for k, v in infer_params.items():
        assert hasattr(ip, k), f"InferParams has no field {k}"
        setattr(ip, k, v)
    return SimpleNamespace(stc = make_stc(tensors, directory), infer_params = ip)
