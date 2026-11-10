// GroupedMoE standalone bench kernel (T59 Experiment 3, follow-up to
// wiki/plans/2026-11-09-grouped-expert-kernel-design.md).
//
// kVer1 kernel shape (deliberately matching the design doc's two-stage structure):
//   - one launch per stage, gate/up stage + down stage (activation fused in
//     the gate/up epilogue), gridded over DISTINCT ROUTED EXPERTS x N-tiles, not
//     over (token, expert) slots;
//   - per CTA, one weight element is loaded once per expert and reused across
//     every row assigned to that expert (MMVQ "one weight load per expert"
//     pattern), as opposed to a per-slot kernel that re-streams the expert's
//     weights every time the expert is picked;
//   - routing is not reimplemented here (cf. exllamav3/exllamav3_ext/routing.cu).
//     The bench driver (grouped_kernel_bench.py) pre-compacts (token, expert)
//     selections into a (distinct experts, row-run table) form on the host and
//     hands them in. The compaction wall cost is TIMED by the driver and
//     printed as compaction_ms per rows value (host-side, outside the
//     pipelined ker events); it is never silently zero.
//
// Production shapes (bench): hidden H=2560, moe_intermediate I=640, down
// output Ho=2560, local expert count E=131 (of 512 total; split 381),
// topk=10. Bench dims are printed by the driver (actual_dims line).
//
// Run-length handling with topk=10: a run (all token rows hitting one expert)
// holds up to rows*topk (=50 at rows=5) slots, which exceeds MAXR (5). The
// kernel processes each run in chunks of at most MAXR rows: for cnt <= MAXR
// (the common case) the behaviour is identical to a single-pass kernel; for
// longer runs the same weights are re-streamed once per chunk (documented;
// correctness is preserved, the extra traffic is part of the measured ker).
//
// Limits of this standalone kernel vs the proposed production kernel (explicitly
// NOT implemented here):
//   - no run-table compaction kernel on-device, no shared-expert merge, no
//     expert-range masking (out-of-range slots contribute nothing because the
//     driver never offers them), no CPU-split flag handoff (moe_handoff.cu);
//   - quantized trellis/suh/svh weights and Hadamard rotations are replaced by
//     dense fp16 weights and a plain SiLU gate; the 256-wide tile and 512
//     threads of exl3_moe_coop_kernel.cuh are replaced by 8-wide tiles /
//     128 threads so the rows=1 case has enough CTAs to fill the GPU and
//     enough outstanding weight loads per SM to approach DRAM bandwidth;
//   - the real coop kernel (exl3_moe_coop_launch) cannot be driven from this
//     standalone bench (no wheel / .pyd integration per brief); the verdict
//     basis is the T58-measured ker medians only (ROWS=1 0.061 ms,
//     ROWS=5 0.117 ms) and the pre-registered pass marks (ROWS=1 <=0.061,
//     ROWS=5 <=0.082). This is printed by the driver as
//     coop_direct_drive: unavailable.
//
// CUDA only, no custom autograd: plain tensor-in / tensor-out pybind calls.
// Pure CUDA TU (no torch headers): including torch here pulls CCCL 13.x +
// Windows SDK headers into cudafe++ on this box; the torch-facing glue lives
// in grouped_glue.cpp and calls the launchers declared in grouped_kernel.h.

#include "grouped_kernel.h"

#define TILE_N 8           // N columns per CTA (I % 8 == 0, Ho % 8 == 0)
#define NTHREADS 128       // 4 warps per CTA
#define NWARP (NTHREADS / 32)
#define COLS_PER_WARP (TILE_N / NWARP)   // sequential N outputs per warp
#define MAXR 5             // max token rows per chunk; longer runs are chunked

// Warp-wide sum reduction of the gate/up accumulators over K-split lanes.
__device__ __forceinline__ void warp_reduce_pair(float a[MAXR], float b[MAXR])
{
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        #pragma unroll
        for (int r = 0; r < MAXR; ++r)
        {
            a[r] += __shfl_down_sync(0xffffffffu, a[r], off);
            b[r] += __shfl_down_sync(0xffffffffu, b[r], off);
        }
    }
}

// Stage A: gate/up GEMVs grouped by distinct expert + SiLU epilogue.
//
// Block (bx, by): bx = distinct-index, by = N-tile index (TILE_N columns).
// The warp owning output column n splits the K reduction across its 32 lanes
// (half2 pairs, coalesced), and every weight element fetched serves all rows
// of the expert's chunk (MMVQ grouped-GEMV: one weight load per expert per
// chunk). Runs longer than MAXR are processed in successive chunks, each
// re-streaming the weights (correctness for topk=10; see file header).
__global__ __launch_bounds__(NTHREADS) void stageA_kernel(
    const half* __restrict__ x,            // (R, H)  fp16
    const half* __restrict__ wg,           // (E, I, H) fp16
    const half* __restrict__ wu,           // (E, I, H) fp16
    const int32_t* __restrict__ dist_exp,  // (D) distinct expert ids
    const int32_t* __restrict__ seg_start, // (D) run start in row_perm
    const int32_t* __restrict__ seg_cnt,   // (D) run length (up to R*topk)
    const int32_t* __restrict__ row_perm,  // (R*topk) token row per slot, run-major
    float* __restrict__ act,               // (R*topk, I) fp32 per-slot, SiLU(gate) * up
    const int H,
    const int I)
{
    const int e = dist_exp[blockIdx.x];
    const int s0 = seg_start[blockIdx.x];
    const int cnt = seg_cnt[blockIdx.x];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n_base = (int)blockIdx.y * TILE_N;
    const int64_t WStride = (int64_t)I * H;

    const half* Wgp = wg + (int64_t)e * WStride;
    const half* Wup = wu + (int64_t)e * WStride;

    // Process the run in chunks of at most MAXR rows. A single chunk
    // (cnt <= MAXR) is the pre-existing single-pass behaviour.
    for (int base = 0; base < cnt; base += MAXR)
    {
        const int remain = cnt - base;
        const int cl = (remain < MAXR) ? remain : MAXR;

        // Hoisted per-chunk row lookups: the chunk's rows depend only on the
        // expert, not on the column or the K step.
        const half2* x2[MAXR];
        #pragma unroll
        for (int r = 0; r < MAXR; ++r)
            x2[r] = nullptr;
        #pragma unroll
        for (int r = 0; r < MAXR; ++r)
            if (r < cl)
                x2[r] = reinterpret_cast<const half2*>(x + (int64_t)row_perm[s0 + base + r] * H);

        #pragma unroll
        for (int q = 0; q < COLS_PER_WARP; ++q)
        {
            const int n = n_base + warp * COLS_PER_WARP + q;
            // Keep out-of-range lanes converged for the shuffle reduction.
            const int n_clamped = n < I ? n : (I - 1);

            float ag[MAXR];
            float au[MAXR];
            #pragma unroll
            for (int r = 0; r < MAXR; ++r) { ag[r] = 0.0f; au[r] = 0.0f; }

            const half2* Wg2 = reinterpret_cast<const half2*>(Wgp + (int64_t)n_clamped * H);
            const half2* Wu2 = reinterpret_cast<const half2*>(Wup + (int64_t)n_clamped * H);
            // K/64 half2 steps (H is a multiple of 64): lane l reads the pair
            // (64*j + 2*l, 64*j + 2*l + 1); each half2 is reused for g and u.
            const int ksteps = H / 64;
            #pragma unroll 8
            for (int j = 0; j < ksteps; ++j)
            {
                const int ko2 = 32 * j + lane;   // half2 offset within the row
                const half2 g2 = __ldg(Wg2 + ko2);
                const half2 u2 = __ldg(Wu2 + ko2);
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                {
                    if (r < cl)
                    {
                        const half2 xv = x2[r][ko2];
                        ag[r] = fmaf(__half2float(g2.x), __half2float(xv.x),
                                     fmaf(__half2float(g2.y), __half2float(xv.y), ag[r]));
                        au[r] = fmaf(__half2float(u2.x), __half2float(xv.x),
                                     fmaf(__half2float(u2.y), __half2float(xv.y), au[r]));
                    }
                }
            }
            warp_reduce_pair(ag, au);
            if (lane == 0 && n < I)
            {
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                {
                    if (r < cl)
                    {
                        // act is per-slot (R*topk, I), indexed by the same
                        // sorted-slot position used in stage B: each slot of
                        // a row passes through a different expert, so rows
                        // must not share an activation (topk > 1 safe).
                        const float s = 1.0f / (1.0f + __expf(-ag[r]));
                        act[(int64_t)(s0 + base + r) * I + n] = ag[r] * s * au[r];
                    }
                }
            }
        }
    }
}

// Stage B: down GEMV grouped by distinct expert. With topk == 1 every
// (row, col) has exactly one writer (direct store, scratch need not be
// zeroed); with topk > 1 contributions accumulate via atomicAdd. Long runs
// are chunked like Stage A (re-stream per chunk, disjoint rows, no conflict).
template <bool ACCUM>
__global__ __launch_bounds__(NTHREADS) void stageB_kernel(
    const float* __restrict__ act,         // (R*topk, I) fp32 per-slot
    const half* __restrict__ wd,           // (E, Ho, I) fp16
    const half* __restrict__ rw,           // (R, topk) fp16 routing weights
    const int32_t* __restrict__ dist_exp,
    const int32_t* __restrict__ seg_start,
    const int32_t* __restrict__ seg_cnt,
    const int32_t* __restrict__ row_perm,
    const int32_t* __restrict__ slot_perm, // (R*topk) slot index within row
    float* __restrict__ out,               // (R, Ho) fp32 output
    const int I,
    const int Ho,
    const int topk)
{
    const int e = dist_exp[blockIdx.x];
    const int s0 = seg_start[blockIdx.x];
    const int cnt = seg_cnt[blockIdx.x];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n_base = (int)blockIdx.y * TILE_N;
    const int64_t WStride = (int64_t)Ho * I;

    const half* Wdp = wd + (int64_t)e * WStride;

    for (int base = 0; base < cnt; base += MAXR)
    {
        const int remain = cnt - base;
        const int cl = (remain < MAXR) ? remain : MAXR;

        // Hoisted per-chunk lookups (rows, activations, routing weights).
        const float2* act2[MAXR];
        float rws[MAXR];
        #pragma unroll
        for (int r = 0; r < MAXR; ++r)
        {
            act2[r] = nullptr;
            rws[r] = 0.0f;
        }
        #pragma unroll
        for (int r = 0; r < MAXR; ++r)
            if (r < cl)
            {
                const int row = row_perm[s0 + base + r];
                // act is per-slot (R*topk, I): read back this slot's activation.
                act2[r] = reinterpret_cast<const float2*>(act + (int64_t)(s0 + base + r) * I);
                rws[r] = __half2float(rw[(int64_t)row * topk + slot_perm[s0 + base + r]]);
            }

        #pragma unroll
        for (int q = 0; q < COLS_PER_WARP; ++q)
        {
            const int n = n_base + warp * COLS_PER_WARP + q;
            const int n_clamped = n < Ho ? n : (Ho - 1);

            float acc[MAXR];
            #pragma unroll
            for (int r = 0; r < MAXR; ++r) acc[r] = 0.0f;

            const half2* Wd2 = reinterpret_cast<const half2*>(Wdp + (int64_t)n_clamped * I);
            const int ksteps = I / 64;
            #pragma unroll 8
            for (int j = 0; j < ksteps; ++j)
            {
                const int ko2 = 32 * j + lane;
                const half2 w2 = __ldg(Wd2 + ko2);
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                {
                    if (r < cl)
                    {
                        const float2 av = act2[r][ko2];
                        acc[r] = fmaf(__half2float(w2.x), av.x,
                                      fmaf(__half2float(w2.y), av.y, acc[r]));
                    }
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
            {
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                    acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
            }
            if (lane == 0 && n < Ho)
            {
                #pragma unroll
                for (int r = 0; r < MAXR; ++r)
                {
                    if (r < cl)
                    {
                        const int row = row_perm[s0 + base + r];
                        const float v = rws[r] * acc[r];
                        if (ACCUM)
                            atomicAdd(&out[(int64_t)row * Ho + n], v);
                        else
                            out[(int64_t)row * Ho + n] = v;
                    }
                }
            }
        }
    }
}

extern "C" void grouped_stageA(
    const half* x, const half* wg, const half* wu,
    const int32_t* dist_exp, const int32_t* seg_start, const int32_t* seg_cnt,
    const int32_t* row_perm, float* act,
    int R, int H, int I, int D,
    cudaStream_t stream)
{
    (void)R;
    dim3 grid(D, (unsigned)(I / TILE_N));
    stageA_kernel<<<grid, NTHREADS, 0, stream>>>(
        x, wg, wu, dist_exp, seg_start, seg_cnt, row_perm, act, H, I);
}

extern "C" void grouped_stageB(
    const float* act, const half* wd, const half* rw,
    const int32_t* dist_exp, const int32_t* seg_start, const int32_t* seg_cnt,
    const int32_t* row_perm, const int32_t* slot_perm,
    float* out, int I, int Ho, int topk, int D,
    cudaStream_t stream)
{
    dim3 grid(D, (unsigned)(Ho / TILE_N));
    if (topk == 1)
        stageB_kernel<false><<<grid, NTHREADS, 0, stream>>>(
            act, wd, rw, dist_exp, seg_start, seg_cnt, row_perm, slot_perm,
            out, I, Ho, topk);
    else
        stageB_kernel<true><<<grid, NTHREADS, 0, stream>>>(
            act, wd, rw, dist_exp, seg_start, seg_cnt, row_perm, slot_perm,
            out, I, Ho, topk);
}
