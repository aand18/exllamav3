#pragma once

#include <cstdint>
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

// Persistent-worker handoff for CPU-offloaded MoE experts (pattern follows the native TP
// backend's CPU-assisted all-reduce: pinned shared-memory flags, GPU-side publish/wait kernels,
// a spin-parked child process consuming a job ring).
//
// Shared-memory layout (allocated and cudaHostRegistered by the parent, plain-mapped by the
// child; all offsets in bytes from the region base):
//
//   [ctrl block]
//     u32 quit          @ 0    parent -> child: exit the worker loop
//     u32 pass_wake     @ 64   parent bumps at the start of every forward pass: child spins hard
//     u32 abort         @ 128  set by the GPU wait kernel on timeout; parent checks per pass
//     u32 ready         @ 192  child -> parent: worker loop entered, layers loaded
//     u32 jobs_tail     @ 256  parent producer index (job ring)
//     u32 jobs_head     @ 320  child consumer index
//     MoeJob ring       @ 384  MOE_JOB_RING entries
//     flag data_ready[] @ SLOT_FLAGS_OFFSET       per slot, stride 64: GPU publishes job seq
//     flag done[]       @ SLOT_FLAGS_OFFSET + 64*MOE_MAX_SLOTS, stride 64: child publishes seq
//     flag consumed[]   @ SLOT_FLAGS_OFFSET + 2*64*MOE_MAX_SLOTS, stride 64: the collecting
//                         device's stream publishes seq after reading the slot output back.
//                         Slot reuse waits on this, not done: with offloaded layers spread
//                         over multiple GPUs, the next tenant's issue runs on a different
//                         stream than the previous tenant's readback, and done (child
//                         finished writing) alone lets the child overwrite the output while
//                         the readback is still queued on the other device
//     flag expert_done[] @ SLOT_FLAGS_OFFSET + 3*64*MOE_MAX_SLOTS, stride 64: per-expert
//                         completion (MOE_JOB_KIND_COMPUTE_EXPERT, EXL3_MOE_PARTIAL_COMPLETION).
//                         Indexed by RING POSITION, not by slot: every job of a layer carries
//                         the same seq, and the child publishes expert_done[pos % MOE_JOB_RING]
//                         once that job's output slice, its row index and its segment entry are
//                         written. The collecting stream waits per job and folds in enqueue
//                         (== expert ascending) order, then publishes consumed once per slot
//   [slot data]         @ MOE_CTRL_SIZE, num_slots x slot_size
//     per slot: [x fp16 cap_rows*Hi][sel i32 cap_rows*topk][w fp16 cap_rows*topk]
//               [out fp32 out_rows*Ho], each section 64-byte aligned. out_rows is cap_rows for
//               the whole-layer paths and cap_rows*topk for the per-expert path, which needs one
//               compact slice row per (token, expert) assignment. The per-expert path appends
//               [rowidx i32 out_rows] (token row of each slice row) and [segtab i32 2*E_cpu]
//               ((offset, count) per worker-local expert id) after the out section
//
// Ordering: the parent writes a job descriptor and bumps jobs_tail before enqueueing the GPU
// memcpys and the data_ready publish for that seq, so the child always sees the descriptor
// before it can see the data flag. Sequence numbers increase monotonically from 1; slot reuse is
// safe because the GPU stream serializes on the wait kernel of the oldest outstanding job.
//
// Flag width: the GPU waits for a flag to reach a sequence number with a stream memory operation.
// CUDA's 32-bit wait compares cyclically ((int32_t)(*addr - value) >= 0), so 32-bit flags may wrap.
// HIP's compares as plain unsigned >=, which a wrapped 32-bit flag would satisfy early, so ROCm
// builds use 64-bit flags that never wrap. Either way each flag owns a 64-byte line, and the job
// descriptor carries the sequence number as two 32-bit halves. CPU-side waits compare the low
// halves cyclically, which is exact for any width.
#if defined(USE_ROCM)
#define MOE_FLAG_BITS 64
typedef uint64_t moe_flag_t;
#else
#define MOE_FLAG_BITS 32
typedef uint32_t moe_flag_t;
#endif

#define MOE_JOB_RING 2048
#define MOE_MAX_SLOTS 8
#define MOE_MAX_WSLOTS 8
#define MOE_JOB_KIND_COMPUTE 0
#define MOE_JOB_KIND_STAGE 1
// GATED: issued by the fused-issue kernel, whose collect side reads the output only when at
// least one selected expert was CPU-resident. The worker may skip the compute and the
// output write entirely for an all-inactive job. Plain COMPUTE jobs must keep writing
// (zeroing) the output, since their collect reads it back unconditionally
#define MOE_JOB_KIND_COMPUTE_GATED 2
// PER-EXPERT (EXL3_MOE_PARTIAL_COMPLETION): one job per distinct CPU-routed expert of a layer,
// all sharing the layer's input slot and seq. The parent writes the fixed descriptor fields and
// reserves `rows*topk` consecutive ring positions; the fused issue kernel's announce step then
// fills experts[0] (worker-local expert id, or MOE_EXPERT_SKIP for a reserved position that
// turned out not to be needed) and prev_seq (the expert's slice offset in rows) from the GPU,
// after the staging kernel has written sel and before data_ready is published -- so the worker
// re-reads both only after its data_ready wait. The worker writes the slice, rowidx and
// segtab[2*expert] and publishes expert_done[ring position]
#define MOE_JOB_KIND_COMPUTE_EXPERT 3
// experts[0] of a reserved per-expert position with no job behind it (fewer distinct CPU
// experts than rows*topk)
#define MOE_EXPERT_SKIP 0xFFFFFFFFu

// kind == STAGE: the worker memcpys `rows` experts' packed weights (ids in experts[]) of layer
// `layer` into weight-staging slot `slot`, after waiting for the parent's pinned_free flag to
// reach prev_seq (the slot's previous tenant has been DMA'd to VRAM). Weight-staging slots
// follow the compute slots in the shared region; their geometry travels in the layout dict.
//
// MOE_JOB_MAX_EXPERTS is the structural capacity of the experts[] array (just sizes a uint32_t
// array in shared memory; overprovisioning has no runtime cost since only the first `rows`
// entries are ever read). The runtime batch size actually used per stage job is a separate,
// Python-side tunable (MoeCpuTuning.batch_experts in moe_cpu_host.py, default 24) clamped to
// this capacity.
#define MOE_JOB_MAX_EXPERTS 256

struct MoeJob
{
    uint32_t seq;
    uint32_t layer;
    uint32_t rows;       // compute: token rows; stage: expert count
    uint32_t topk;
    uint32_t slot;
    uint32_t kind;
    uint32_t prev_seq;   // stage: pinned_free value to wait for before overwriting the slot
    uint32_t experts[MOE_JOB_MAX_EXPERTS];
    uint32_t seq_hi;     // high 32 bits of seq (published in full on 64-bit flags)
};

#define MOE_CTRL_JOBS_OFFSET 384
#define MOE_SLOT_FLAGS_OFFSET (MOE_CTRL_JOBS_OFFSET + MOE_JOB_RING * sizeof(MoeJob))

// Flag order: data_ready[MAX_SLOTS], done[MAX_SLOTS], consumed[MAX_SLOTS],
// expert_done[MOE_JOB_RING], stage_done[MAX_WSLOTS], pinned_free[MAX_WSLOTS], each entry
// 64-byte strided. expert_done is ring-indexed (it belongs to a job, not to a slot); the weight
// stager's two banks stay last so their offsets are unaffected by the per-expert bank's growth
#define MOE_FLAGS_SIZE (3 * 64 * MOE_MAX_SLOTS + 64 * MOE_JOB_RING + 2 * 64 * MOE_MAX_WSLOTS)
#define MOE_EXPERT_DONE_OFFSET (MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS)
#define MOE_STAGE_FLAGS_OFFSET (MOE_EXPERT_DONE_OFFSET + 64 * MOE_JOB_RING)

// Stage jobs travel in their own ring, consumed by a dedicated stager thread in the child, so
// weight staging (pure memcpy) overlaps the compute pool's work on the token tail instead of
// queuing behind it
#define MOE_STAGE_RING 64
#define MOE_STAGE_TAIL_OFFSET (MOE_SLOT_FLAGS_OFFSET + MOE_FLAGS_SIZE)
#define MOE_STAGE_HEAD_OFFSET (MOE_STAGE_TAIL_OFFSET + 64)
#define MOE_STAGE_JOBS_OFFSET (MOE_STAGE_TAIL_OFFSET + 128)
#define MOE_CTRL_SIZE (MOE_STAGE_JOBS_OFFSET + MOE_STAGE_RING * sizeof(MoeJob))

// GPU-side flag ops on the current torch CUDA stream (addresses inside the registered region).
// Prefer stream memory operations (front-end executed, no SM occupancy, no launch cost) over the
// kernel fallback; exl3_moe_cpu_set_memops toggles this process-wide (called once, parent-side
// only -- these ops are only ever issued from the process enqueueing the GPU work).
void exl3_moe_flag_write(uintptr_t flag, int64_t value);
void exl3_moe_flag_wait(uintptr_t flag, int64_t value, uintptr_t abort_flag);

// Fused split-path handoff (see moe_handoff.cu): the job descriptor, the slot's consumed wait,
// the issue kernel and the data_ready flag in one call; the done wait, the collect kernel and the
// consumed flag in the other.
//
// `partial` (EXL3_MOE_PARTIAL_COMPLETION) switches the issue side to per-expert jobs: the parent
// has already written `cap` = rows*topk descriptors with kind COMPUTE_EXPERT at ring positions
// partial_tail .. partial_tail+cap-1; this call then only runs the staging kernel and the
// announce kernel (which fills experts[0]/prev_seq of those descriptors from the GPU) and
// publishes data_ready as usual. `e_cpu` bounds the announce kernel's expert histogram. The
// matching collect is per job: exl3_moe_flag_wait on
// expert_done[pos % MOE_JOB_RING], moe_split_collect_add_expert, and ONE flag_write of consumed
// after the last fold -- both issued by the caller (moe_cpu_host.py), so the pairs stay
// stream-ordered without a host sync.
bool exl3_moe_split_issue_fused
(
    uintptr_t shm_base, int64_t seq, int64_t layer, int64_t rows, int64_t topk, int64_t slot, int64_t prev_seq,
    at::Tensor sel, const c10::optional<at::Tensor>& map, const c10::optional<at::Tensor>& hist,
    const at::Tensor& y, const at::Tensor& w, int64_t h_sel_ptr, int64_t h_x_ptr, int64_t h_w_ptr,
    at::Tensor dev_count, int64_t hi, int64_t first_cpu,
    uintptr_t data_ready_flag, uintptr_t consumed_flag, uintptr_t abort_flag,
    int64_t partial, int64_t partial_tail, int64_t cap, int64_t e_cpu
);
void exl3_moe_split_collect_fused
(
    at::Tensor final_out, int64_t h_out_ptr, const at::Tensor& dev_count, int64_t slot, int64_t ho,
    uintptr_t done_flag, int64_t seq, uintptr_t consumed_flag, uintptr_t abort_flag
);
void exl3_moe_cpu_set_memops(bool enabled);

// Child worker loop: consumes jobs until quit is set. Layer indices in job descriptors refer to
// the registration order of exl3_moe_cpu_make_layer calls made in this (child) process.
void exl3_moe_cpu_worker_run
(
    uintptr_t shm_base,
    int64_t num_slots,
    int64_t slot_size,
    int64_t cap_rows,
    int64_t max_hi,
    int64_t max_ho,
    int64_t max_topk,
    int64_t wstage_offset,
    int64_t num_wslots,
    int64_t wslot_size,
    int64_t threads,
    int64_t stage_threads,
    int64_t partial
);
