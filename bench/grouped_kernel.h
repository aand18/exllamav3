#pragma once

// Shared launcher declarations for the standalone grouped-MoE bench kernel.
// Pure CUDA declarations (no torch): grouped_kernel.cu defines these, the
// torch glue TU (grouped_glue.cpp) calls them.
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

// Stage A: gate/up GEMVs grouped by distinct expert, SiLU epilogue fused.
// Grids (D, I / TILE_N), blockDim 128.
void grouped_stageA(
    const half* x,            // (R, H)  fp16
    const half* wg,           // (E, I, H) fp16
    const half* wu,           // (E, I, H) fp16
    const int32_t* dist_exp,  // (D) distinct expert ids
    const int32_t* seg_start, // (D) run start into row_perm
    const int32_t* seg_cnt,   // (D) run length (up to R*topk; kernel chunks >MAXR)
    const int32_t* row_perm,  // (R*topk) token rows, run-major
    float* act,               // (R*topk, I) fp32 per-slot output
    int R, int H, int I, int D,
    cudaStream_t stream);

// Stage B: down GEMV grouped by distinct expert; atomicAdd into `out`.
// act is per-slot (R*topk, I); out must be zero-initialized when topk > 1
// (the kernel accumulates into it).
void grouped_stageB(
    const float* act,         // (R*topk, I) fp32 per-slot
    const half* wd,           // (E, Ho, I) fp16
    const half* rw,           // (R, topk) fp16
    const int32_t* dist_exp,
    const int32_t* seg_start,
    const int32_t* seg_cnt,   // (D) run length (up to R*topk; kernel chunks >MAXR)
    const int32_t* row_perm,
    const int32_t* slot_perm, // (R*topk) slot index within row
    float* out,               // (R, Ho) fp32: written by value at topk==1,
                              // accumulated (needs zero-init) at topk>1
    int I, int Ho, int topk, int D,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
