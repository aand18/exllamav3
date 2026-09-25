# KVarN: 4090 (sm_89) machine handoff

This branch (`wip/kvarn-cache`) is CPU-complete and needs a CUDA box for:
Triton parity acceptance, micro-KLD numbers, and any perf work. Everything
below was verified up to the hardware boundary on a GPU-less box.

## Clone

```sh
git clone https://github.com/aand18/exllamav3
cd exllamav3
git checkout wip/kvarn-cache
```

## Prerequisites

- Python 3.10–3.13, ~15 GB free (CUDA torch ~2 GB + build tree).
- CUDA toolkit 12.4+ (13.x works; cu130 torch + 13.4 toolkit built clean).
- Windows: MSVC 14.x + Windows 10/11 SDK ("Desktop development with C++",
  without the optional clang component — nvcc only accepts MSVC anyway).
- Target arch is sm_89 (RTX 4090) only; keep every other arch out for speed.

## Build the extension

Python files need no build; only `exllamav3/exllamav3_ext/` does. Build in
a venv, never in system Python:

```sh
python -m venv exl-cuda
exl-cuda/Scripts/activate            # or source exl-cuda/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install pydantic tokenizers safetensors numpy rich typing_extensions \
    pillow pyyaml marisa_trie llguidance pytest
```

Windows trap we hit: `VsDevCmd.bat` exports an environment that does NOT
reliably propagate to child shells (ninja/`cl`/`nvcc` then fail with
`CreateProcess` / `Cannot find compiler 'cl.exe'`). Bypass it — set the
toolchain paths manually before building:

```bat
set MSVC=C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.51.36231
set KIT=C:/Program Files (x86)/Windows Kits/10
set PATH=%MSVC%/bin/Hostx64/x64;%KIT%/bin/10.0.26100.0/x64;<cuda-bin>;<venv-Scripts>;%PATH%
set INCLUDE=%MSVC%/include;%KIT%/Include/10.0.26100.0/ucrt;%KIT%/Include/10.0.26100.0/um;%KIT%/Include/10.0.26100.0/shared;%KIT%/Include/10.0.26100.0/winrt;%KIT%/Include/10.0.26100.0/cppwinrt
set LIB=%MSVC%/lib/x64;%KIT%/Lib/10.0.26100.0/ucrt/x64;%KIT%/Lib/10.0.26100.0/um/x64
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.9
set MAX_JOBS=4
python setup.py build_ext --inplace
```

Notes: `TORCH_CUDA_ARCH_LIST` is space-separated (`8.6 9.0+PTX` style —
semicolons break torch's parser). `MAX_JOBS=4` avoids nvcc OOM on 16 GB
RAM; scale to the box. Adjust the MSVC/SDK version numbers to whatever is
installed. Linux: same torch install, then
`TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace`
(or `pip install --no-build-isolation .`).

## Validate, in order

1. CPU-style suite (also runs on CUDA box, torch path is the default):
   `python -m pytest tests/test_kvarn_cpu.py tests/test_kvarn_tail_cpu.py tests/test_kvarn_widths_cpu.py tests/test_kvarn_m4_cpu.py tests/test_kvarn_m5_cpu.py tests/test_kvarn_triton.py -q`
2. Triton acceptance (the kernel has NEVER launched — this is its first run):
   `EXL3_KVARN_TRITON=1 EXL3_KVARN_TRITON_PARITY=1 python -m pytest tests/test_kvarn_triton.py tests/test_kvarn_cpu.py -q`
   Parity mode runs torch + Triton side by side and asserts equality.
   Do not trust `EXL3_KVARN_TRITON=1` without parity passing first.
3. Micro-KLD (needs ~8 GB download, one-time):
   ```sh
   pip install huggingface_hub
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='turboderp/Qwen3.8-27B-exl3', revision='SC_1.40bpw_H3_V3', local_dir='models/Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3', allow_patterns=['model.safetensors','*.json','*.txt','merges.txt','vocab.json','*.jinja'])"
   python eval/kvarn_microkld.py -m models/Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3 -cq kvarn4 -ntok 200
   python eval/kvarn_microkld.py -m models/Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3 -cq kvarn5,kvarn4 -ntok 200
   ```
   Note: that checkpoint is Qwen3.5-dense (`head_dim` 256, 16 full-attention
   layers) — good KVarN coverage, but it exercises neither QSA nor MoE.

## Report back

Parity pass/fail (+ assertion text on failure), micro-KLD median/mean/max +
same-top % per preset, prefill tok/s for fp16 vs kvarn4 vs kvarn5,kvarn4,
and GPU model. Post results to the branch's draft PR (#2).
