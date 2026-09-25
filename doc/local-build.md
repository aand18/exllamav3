# Local build notes (this machine, GPU-less; CUDA target is sm_89 elsewhere)

Learned the hard way — kept here so nobody rediscovers it. Version numbers
are examples "as of Sep 2026", not mandates; adapt to what's installed.

## When a rebuild is needed at all

- Pure Python + Triton changes need NO build (Triton JITs at runtime).
- A built `exllamav3_ext` `.pyd` keeps working until `exllamav3/exllamav3_ext/`
  C++/CUDA sources change. Only then rebuild.
- Never build in the working checkout: copy the tree (minus `.git`,
  `build/`, `*.pyd/obj`) to a scratch dir and build there, so artifacts
  can't leak into commits.

## Windows CUDA build that works (MSVC + nvcc, no GPU on box)

- torch from the CUDA wheel matching the toolkit major
  (e.g. cu130 torch + 13.4 toolkit built clean despite the minor skew).
- `VsDevCmd.bat` exports an environment that does NOT reliably propagate to
  child shells here (ninja/`cl`/`nvcc` then fail with `CreateProcess` /
  `Cannot find compiler 'cl.exe'`). Bypass: set MSVC + SDK `PATH` /
  `INCLUDE` / `LIB` manually (e.g. `.../MSVC/14.51.36231/bin/Hostx64/x64`,
  Windows Kits `10.0.26100.0`), plus `DISTUTILS_USE_SDK=1`.
- `TORCH_CUDA_ARCH_LIST` is space-separated (`8.6 9.0+PTX` style);
  semicolons break torch's parser. sm_89-only (`8.9`) halves nvcc work.
- `MAX_JOBS=4` avoids nvcc OOM on 16 GB RAM; scale to the box.
- Ninja may or may not cooperate; distutils fallback is slow but works.
  Log to a file (`> build.log 2>&1`) — per-TU SDK-header warnings flood.

## No-GPU caveat (transfers to every merged tree too)

A merged tree compiles the same way, but runtime interaction between
features can only be proven on the CUDA box. Mark such test results
CPU-partial. `ext` has no CPU forward kernels (e.g. `rms_norm` is
CUDA-only), so no model-level run is possible on this machine at all.
