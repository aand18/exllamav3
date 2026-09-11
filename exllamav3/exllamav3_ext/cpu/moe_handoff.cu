#include "moe_handoff.h"
#include "moe_mul1.h"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "../ptx.cuh"

#ifdef __linux__
#include <dlfcn.h>
#else
#include <intrin.h>
#include <windows.h>
#endif

// Wait timeout: generous, the CPU may be chewing a full prefill chunk
#define MOE_WAIT_TIMEOUT_NS 30000000000ull
#define MOE_SLEEP_MIN 64
#define MOE_SLEEP_MAX 2048

// -------------------------------------------------------------------------------------------
//   GPU flag kernels
// -------------------------------------------------------------------------------------------

// No longer used
__global__ void moe_flag_write_kernel(uint32_t* flag, uint32_t value)
{
    stg_release_sys_u32(flag, value);
}

// No longer used
__global__ void moe_flag_wait_kernel(uint32_t* flag, uint32_t value, uint32_t* abort_flag)
{
    uint64_t sleep = MOE_SLEEP_MIN;
    uint64_t waited = 0;
    while (true)
    {
        uint32_t v = (uint32_t) ldg_acquire_sys_u32(flag);
        if ((int32_t)(v - value) >= 0) return;
        __nanosleep(sleep);
        waited += sleep;
        if (sleep < MOE_SLEEP_MAX) sleep <<= 1;
        if (waited > MOE_WAIT_TIMEOUT_NS)
        {
            stg_release_sys_u32(abort_flag, 1);
            return;
        }
    }
}

// Stream memory operations: same semantics as the flag kernels (CU_STREAM_WAIT_VALUE_GEQ is
// documented as the cyclic comparison (int32_t)(*addr - value) >= 0, identical to the kernel's
// predicate), but executed by the GPU front-end: no SM occupancy (nothing to co-schedule against
// exl3_moe's all-blocks-resident launch) and no kernel-launch cost per wait. The kernel path
// remains as a fallback (EXL3_MOE_MEMOPS=0 forces it); note the memop wait has no timeout, so
// dead-worker detection moves to the host-side watchdog, which unblocks pending waits by
// writing satisfying values into the flags.
//
// Windows/WDDM note: on WDDM the memop path measured ~10% slower than the kernel fallback
// (36 offloaded layers x every decode step). Two compounding causes, each env-gated:
//  * Visibility (EXL3_MOE_MEMOPS_FLUSH=0 disables, default on): a front-end write to mapped
//    host memory is not guaranteed promptly visible to the CPU worker, nor are the worker's
//    prior writes guaranteed visible downstream of a GPU wait, without an explicit flush.
//    The write is therefore enqueued as a single 2-op batch [write32 + flush-remote-writes
//    barrier] and waits carry WAIT_VALUE_FLUSH -- same number of driver calls as before.
//    Requires CAN_FLUSH_REMOTE_WRITES (queried once per device); without it the code
//    degrades to plain single ops.
//  * Submission (EXL3_MOE_MEMOPS_SUBMIT=0 disables, default on): WDDM batches small ops in
//    a software queue that only drains on heavier calls. The whole-layer decode handshake
//    enqueues no kernel at all (memcpys + memops only), so a write can sit unsubmitted
//    across the Python gap between issue and collect: the worker starts late (late
//    data_ready) and the GPU polls late (late wait). After every memop write we record +
//    query a dummy event, which pushes the software queue without any host sync, at the
//    cost of two cheap driver calls per write.
// EXL3_MOE_MEMOPS_LOG=1 prints a one-time resolution/support line plus any fallback-latch
// trip. A batch/flush-only failure never latches to the kernel fallback by itself: the op
// is retried plain first, and only a plain-op failure latches (preserving the invariant
// that a latched MEMOPS=1 behaves exactly like MEMOPS=0).
namespace {

typedef CUresult (CUDAAPI* fn_stream_wait32)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
typedef CUresult (CUDAAPI* fn_stream_write32)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
typedef CUresult (CUDAAPI* fn_dev_attr)(int*, CUdevice_attribute, CUdevice);
typedef CUresult (CUDAAPI* fn_ctx_dev)(CUdevice*);
// Flush primitives (batch write+flush barrier, WAIT_VALUE_FLUSH, CAN_FLUSH_REMOTE_WRITES
// query) need CUDA 12+ headers. They are enum constants, not macros, so they cannot be
// probed with #if defined(); older toolkits degrade to plain single ops instead.
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
#define MEMOPS_HAVE_FLUSH 1
typedef CUresult (CUDAAPI* fn_stream_batch)(CUstream, unsigned, CUstreamBatchMemOpParams*, unsigned);
#define MEMOPS_BATCH_SYM_V "cuStreamBatchMemOp_v2"
#define MEMOPS_BATCH_STRUCT CUstreamBatchMemOpParams
#endif

struct MemOps
{
    fn_stream_wait32 wait = nullptr;
    fn_stream_write32 write = nullptr;
#ifdef MEMOPS_HAVE_FLUSH
    fn_stream_batch batch = nullptr;
#endif
    fn_dev_attr dev_attr = nullptr;
    fn_ctx_dev ctx_dev = nullptr;
    bool resolved = false;
    MemOps()
    {
        // Symbol resolution is unconditional (independent of exl3_moe_cpu_set_memops): whether
        // the ops are used is a separate, mutable runtime switch, not a one-time decision.
        // The batch entry is the v2 ABI paired with the v2 struct from the same headers;
        // a missing entry degrades to plain ops, so a mismatch can never be called.
#ifdef __linux__
        void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (!h) h = dlopen("libcuda.so", RTLD_LAZY | RTLD_NOLOAD);
        if (!h) return;
        wait = (fn_stream_wait32) dlsym(h, "cuStreamWaitValue32_v2");
        if (!wait) wait = (fn_stream_wait32) dlsym(h, "cuStreamWaitValue32");
        write = (fn_stream_write32) dlsym(h, "cuStreamWriteValue32_v2");
        if (!write) write = (fn_stream_write32) dlsym(h, "cuStreamWriteValue32");
#ifdef MEMOPS_HAVE_FLUSH
        batch = (fn_stream_batch) dlsym(h, MEMOPS_BATCH_SYM_V);
#endif
        dev_attr = (fn_dev_attr) dlsym(h, "cuDeviceGetAttribute");
        ctx_dev = (fn_ctx_dev) dlsym(h, "cuCtxGetDevice");
#else
        HMODULE h = GetModuleHandleA("nvcuda.dll");
        if (!h) return;
        wait = (fn_stream_wait32) GetProcAddress(h, "cuStreamWaitValue32_v2");
        if (!wait) wait = (fn_stream_wait32) GetProcAddress(h, "cuStreamWaitValue32");
        write = (fn_stream_write32) GetProcAddress(h, "cuStreamWriteValue32_v2");
        if (!write) write = (fn_stream_write32) GetProcAddress(h, "cuStreamWriteValue32");
#ifdef MEMOPS_HAVE_FLUSH
        batch = (fn_stream_batch) GetProcAddress(h, MEMOPS_BATCH_SYM_V);
#endif
        dev_attr = (fn_dev_attr) GetProcAddress(h, "cuDeviceGetAttribute");
        ctx_dev = (fn_ctx_dev) GetProcAddress(h, "cuCtxGetDevice");
#endif
        resolved = wait && write;
    }
};

MemOps& memops() { static MemOps m; return m; }
std::atomic<bool> g_memops_ok { true };
std::atomic<bool> g_memops_enabled { true };
std::atomic<bool> g_memops_logged { false };
#ifdef MEMOPS_HAVE_FLUSH
std::atomic<bool> g_memops_batch_ok { true };       // cleared on first batch-only failure
#endif
#ifdef MEMOPS_HAVE_FLUSH
std::atomic<bool> g_memops_wait_flush_ok { true };  // cleared on first FLUSH-wait failure
#endif

struct MemOpsCfg
{
    bool flush = true;   // EXL3_MOE_MEMOPS_FLUSH (default 1)
    bool submit = true;  // EXL3_MOE_MEMOPS_SUBMIT (default 1)
    bool log = false;    // EXL3_MOE_MEMOPS_LOG (default 0)
};

MemOpsCfg& memops_cfg()
{
    static MemOpsCfg c;
    static std::once_flag once;
    std::call_once(once, []{
        if (const char* e = std::getenv("EXL3_MOE_MEMOPS_FLUSH")) c.flush = e[0] != '0';
        if (const char* e = std::getenv("EXL3_MOE_MEMOPS_SUBMIT")) c.submit = e[0] != '0';
        c.log = std::getenv("EXL3_MOE_MEMOPS_LOG") != nullptr;
    });
    return c;
}

void memops_latch_fallback(const char* what)
{
    bool expected = true;
    if (g_memops_ok.compare_exchange_strong(expected, false) && memops_cfg().log)
        std::fprintf(stderr, "[exl3][memops] %s failed, latching to kernel fallback\n", what);
}

// Device owning the calling thread's current context (-1 when it cannot be determined;
// callers then treat flush as unsupported, which is always safe: plain ops are correct,
// just potentially slower).
int memops_current_dev(MemOps& m)
{
    CUdevice d = 0;
    if (m.ctx_dev && m.ctx_dev(&d) == CUDA_SUCCESS && d >= 0) return (int) d;
    return -1;
}

// CAN_FLUSH_REMOTE_WRITES, queried once per device (attribute queries are driver calls and
// must not run per flag op). Benign races: concurrent first-use queries on two devices may
// repeat a query; the cached pair is only ever (dev, support-for-dev).
bool memops_can_flush(MemOps& m, int dev)
{
    static std::atomic<int> cached_dev { -2 };
    static std::atomic<bool> cached_sup { false };
    if (dev >= 0 && cached_dev.load(std::memory_order_relaxed) == dev)
        return cached_sup.load(std::memory_order_relaxed);
    bool sup = false;
#ifdef MEMOPS_HAVE_FLUSH
    if (m.dev_attr && dev >= 0)
    {
        int v = 0;
        if (m.dev_attr(&v, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, (CUdevice) dev)
            == CUDA_SUCCESS && v)
            sup = true;
    }
#endif
    if (dev >= 0)
    {
        cached_sup.store(sup, std::memory_order_relaxed);
        cached_dev.store(dev, std::memory_order_relaxed);
    }
    return sup;
}

void memops_log_once(MemOps& m)
{
    bool expected = false;
    if (!g_memops_logged.compare_exchange_strong(expected, true)) return;
    MemOpsCfg& c = memops_cfg();
    int dev = memops_current_dev(m);
    std::fprintf(stderr,
        "[exl3][memops] wait/write %s, batch %s, can_flush_remote_writes(dev %d) %d, "
        "flush %d, submit %d, enabled %d\n",
        (m.wait && m.write) ? "ok" : "MISSING",
#ifdef MEMOPS_HAVE_FLUSH
        m.batch ? "ok" : "missing",
#else
        "unsupported-by-build",
#endif
        dev, (dev >= 0 && memops_can_flush(m, dev)) ? 1 : 0,
        c.flush ? 1 : 0, c.submit ? 1 : 0,
        g_memops_enabled.load(std::memory_order_relaxed) ? 1 : 0);
}

// Push the WDDM software queue without any host sync: the record enqueues a marker behind
// the write, and the query forces the driver to submit the queued work to answer it. The
// event is never waited on or synchronized; a cross-device record error is ignored by
// design (the push is best-effort, correctness never depends on it). One cached event per
// device; creation is locked, record/query are thread-safe.
void memops_push_submit(cudaStream_t stream)
{
    static cudaEvent_t evs[8] = {};
    static std::mutex mtx;
    int dev = 0;
    if (::cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 8) return;
    cudaEvent_t ev = evs[dev];
    if (!ev)
    {
        std::lock_guard<std::mutex> lk(mtx);
        ev = evs[dev];
        if (!ev)
        {
            if (::cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return;
            evs[dev] = ev;
        }
    }
    if (::cudaEventRecord(ev, stream) != cudaSuccess) return;
    (void) ::cudaEventQuery(ev);  // NotReady until the stream drains: expected, ignored
}

// Enqueue one flag write via memops. Returns true when the write is on the stream (caller
// then optionally pushes submission); false selects the kernel fallback (latching first
// when even the plain op failed).
bool memops_write(CUstream stream, CUdeviceptr flag, cuuint32_t value)
{
    MemOps& m = memops();
    MemOpsCfg& c = memops_cfg();
    if (c.log) memops_log_once(m);
#ifdef MEMOPS_HAVE_FLUSH
    if (c.flush && m.batch && g_memops_batch_ok.load(std::memory_order_relaxed)
        && memops_can_flush(m, memops_current_dev(m)))
    {
        MEMOPS_BATCH_STRUCT p[2];
        std::memset(p, 0, sizeof(p));
        p[0].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
        p[0].writeValue.address = flag;
        p[0].writeValue.value = value;
        p[0].writeValue.flags = 0;
        p[1].flushRemoteWrites.operation = CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES;
        p[1].flushRemoteWrites.flags = 0;
        if (m.batch(stream, 2, p, 0) == CUDA_SUCCESS) return true;
        // Flush-only failure (e.g. no HW support on this device): stop trying the batch,
        // retry plain before ever latching -- a flush problem must not disable memops.
        g_memops_batch_ok.store(false, std::memory_order_relaxed);
        if (c.log)
            std::fprintf(stderr, "[exl3][memops] batch write+flush rejected, using plain writes\n");
    }
#endif
    if (m.write(stream, flag, value, 0) == CUDA_SUCCESS) return true;
    memops_latch_fallback("write");
    return false;
}

// Enqueue one flag wait via memops. Same latch contract as memops_write.
bool memops_wait(CUstream stream, CUdeviceptr flag, cuuint32_t value)
{
    MemOps& m = memops();
    MemOpsCfg& c = memops_cfg();
    if (c.log) memops_log_once(m);
#ifdef MEMOPS_HAVE_FLUSH
    if (c.flush && g_memops_wait_flush_ok.load(std::memory_order_relaxed)
        && memops_can_flush(m, memops_current_dev(m)))
    {
        if (m.wait(stream, flag, value,
                   (unsigned)(CU_STREAM_WAIT_VALUE_GEQ | CU_STREAM_WAIT_VALUE_FLUSH))
            == CUDA_SUCCESS)
            return true;
        g_memops_wait_flush_ok.store(false, std::memory_order_relaxed);
        if (c.log)
            std::fprintf(stderr, "[exl3][memops] FLUSH wait rejected, using plain waits\n");
    }
#endif
    if (m.wait(stream, flag, value, CU_STREAM_WAIT_VALUE_GEQ) == CUDA_SUCCESS) return true;
    memops_latch_fallback("wait");
    return false;
}

} // namespace

void exl3_moe_cpu_set_memops(bool enabled)
{
    g_memops_enabled.store(enabled, std::memory_order_relaxed);
}

void exl3_moe_flag_write(uintptr_t flag, int64_t value)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    MemOps& m = memops();
    if (m.resolved && g_memops_enabled.load(std::memory_order_relaxed)
        && g_memops_ok.load(std::memory_order_relaxed))
    {
        if (memops_write((CUstream) stream, (CUdeviceptr) flag, (cuuint32_t) value))
        {
            if (memops_cfg().submit) memops_push_submit(stream);
            return;
        }
    }
    moe_flag_write_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint32_t*>(flag), static_cast<uint32_t>(value));
}

void exl3_moe_flag_wait(uintptr_t flag, int64_t value, uintptr_t abort_flag)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    MemOps& m = memops();
    if (m.resolved && g_memops_enabled.load(std::memory_order_relaxed)
        && g_memops_ok.load(std::memory_order_relaxed))
    {
        if (memops_wait((CUstream) stream, (CUdeviceptr) flag, (cuuint32_t) value)) return;
    }
    moe_flag_wait_kernel<<<1, 1, 0, stream>>>
    (
        reinterpret_cast<uint32_t*>(flag),
        static_cast<uint32_t>(value),
        reinterpret_cast<uint32_t*>(abort_flag)
    );
}

// ---- child worker loop --------------------------------------------------------------------

namespace {

inline void cpu_pause_()
{
#ifdef __linux__
    __builtin_ia32_pause();
#else
    _mm_pause();
#endif
}

inline uint32_t load_acquire_u32(const uint32_t* p)
{
#if defined(_MSC_VER) && !defined(__clang__)
    long v = _InterlockedCompareExchange(
        reinterpret_cast<volatile long*>(const_cast<uint32_t*>(p)), 0L, 0L);
    return static_cast<uint32_t>(v);
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

inline void store_release_u32(uint32_t* p, uint32_t v)
{
#if defined(_MSC_VER) && !defined(__clang__)
    (void)_InterlockedExchange(reinterpret_cast<volatile long*>(p), static_cast<long>(v));
#else
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

inline size_t align64(size_t x) { return (x + 63) & ~size_t(63); }

} // namespace

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
    int64_t stage_threads
)
{
    // GIL is released by the binding's call_guard; do not release it again here
    uint8_t* base = reinterpret_cast<uint8_t*>(shm_base);
    uint32_t* quit = reinterpret_cast<uint32_t*>(base + 0);
    uint32_t* pass_wake = reinterpret_cast<uint32_t*>(base + 64);
    uint32_t* ready = reinterpret_cast<uint32_t*>(base + 192);
    uint32_t* jobs_tail = reinterpret_cast<uint32_t*>(base + 256);
    uint32_t* jobs_head = reinterpret_cast<uint32_t*>(base + 320);
    MoeJob* jobs = reinterpret_cast<MoeJob*>(base + MOE_CTRL_JOBS_OFFSET);
    uint32_t* data_ready = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET);
    uint32_t* done = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 64 * MOE_MAX_SLOTS);
    uint32_t* stage_done = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS);
    uint32_t* pinned_free = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS + 64 * MOE_MAX_WSLOTS);
    uint32_t* stage_tail = reinterpret_cast<uint32_t*>(base + MOE_STAGE_TAIL_OFFSET);
    uint32_t* stage_head = reinterpret_cast<uint32_t*>(base + MOE_STAGE_HEAD_OFFSET);
    MoeJob* stage_jobs = reinterpret_cast<MoeJob*>(base + MOE_STAGE_JOBS_OFFSET);
    uint8_t* data = base + MOE_CTRL_SIZE;
    uint8_t* wstage = base + wstage_offset;

    // Dedicated stager: consumes the stage ring so weight memcpys run concurrently with the
    // compute pool's work on the token tail. Uses its own scratch threads, never the pool.
    // stage_threads comes from the parent (MoeCpuTuning), passed through the "start" layout.
    int stage_threads_ = stage_threads > 0 ? (int) stage_threads : 1;
    std::thread stager([&]()
    {
        uint32_t shead = load_acquire_u32(stage_head);
        uint32_t s_last_wake = 0;
        int s_idle = 0;
        while (true)
        {
            if (load_acquire_u32(quit)) return;
            const uint32_t wake = load_acquire_u32(pass_wake);
            if (wake != s_last_wake) { s_last_wake = wake; s_idle = 0; }
            if (load_acquire_u32(stage_tail) == shead)
            {
                if (++s_idle < 65536) { cpu_pause_(); continue; }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            s_idle = 0;

            const MoeJob job = stage_jobs[shead % MOE_STAGE_RING];
            shead++;
            store_release_u32(stage_head, shead);

            // Wait until the slot's previous tenant has been DMA'd out
            uint32_t* pf = pinned_free + size_t(job.slot) * 16;
            while ((int32_t)(load_acquire_u32(pf) - job.prev_seq) < 0)
            {
                if (load_acquire_u32(quit)) return;
                cpu_pause_();
            }
            exl3_moe_cpu_stage_experts(
                static_cast<int64_t>(job.layer),
                job.experts,
                static_cast<int>(job.rows),
                wstage + size_t(job.slot) * wslot_size,
                stage_threads_
            );
            store_release_u32(stage_done + size_t(job.slot) * 16, job.seq);
        }
    });

    // Fixed slot section offsets from the registered maxima
    const size_t off_x = 0;
    const size_t off_sel = align64(off_x + size_t(cap_rows) * max_hi * 2);
    const size_t off_w = align64(off_sel + size_t(cap_rows) * max_topk * 4);
    const size_t off_out = align64(off_w + size_t(cap_rows) * max_topk * 2);

    // Handoff profiling (EXL3_MOE_HANDOFF_PROF): per-job wall times for the three segments the
    // worker can observe -- idle gap since the previous job, spin-on-data_ready, and compute
    const bool hprof = getenv("EXL3_MOE_HANDOFF_PROF") != nullptr;
    double hp_gap = 0.0, hp_spin = 0.0, hp_comp = 0.0;
    double hp_gap_mx = 0.0, hp_spin_mx = 0.0, hp_comp_mx = 0.0;
    long hp_jobs = 0, hp_empty = 0, hp_assign = 0, hp_rows = 0;
    auto hp_prev_end = std::chrono::steady_clock::now();

    store_release_u32(ready, 1);

    uint32_t head = load_acquire_u32(jobs_head);
    uint32_t last_wake = 0;
    int idle = 0;
    while (true)
    {
        if (load_acquire_u32(quit)) break;

        const uint32_t wake = load_acquire_u32(pass_wake);
        if (wake != last_wake) { last_wake = wake; idle = 0; }

        if (load_acquire_u32(jobs_tail) == head)
        {
            // Spin hard in-pass (jobs arrive within microseconds of the GPU reaching the layer),
            // back off to naps when the queue has been dry for a while
            if (++idle < 65536) { cpu_pause_(); continue; }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        idle = 0;

        const MoeJob job = jobs[head % MOE_JOB_RING];
        head++;
        store_release_u32(jobs_head, head);

        const auto hp_t0 = std::chrono::steady_clock::now();

        // Wait for the GPU to publish the staged inputs for this seq
        uint32_t* drdy = data_ready + size_t(job.slot) * 16;
        while ((int32_t)(load_acquire_u32(drdy) - job.seq) < 0)
        {
            if (load_acquire_u32(quit)) goto out;
            cpu_pause_();
        }

        const auto hp_t1 = std::chrono::steady_clock::now();

        {
            uint8_t* slot = data + size_t(job.slot) * slot_size;
            bool run = true;
            if (job.kind == MOE_JOB_KIND_COMPUTE_GATED)
            {
                // Fused-issue job: the collecting kernel reads the output only when some
                // selected expert is CPU-resident, so an all-inactive job is a pure no-op
                const int32_t* selp = reinterpret_cast<const int32_t*>(slot + off_sel);
                const int total = (int) job.rows * (int) job.topk;
                if (hprof)
                {
                    // Full-count variant: CPU-assignment stats for the profiler report
                    int n = 0;
                    for (int i = 0; i < total; ++i)
                        if (selp[i] >= 0) n++;
                    run = n > 0;
                    hp_assign += n;
                    hp_rows += (int) job.rows;
                    if (!run) hp_empty++;
                }
                else
                {
                    run = false;
                    for (int i = 0; i < total; ++i)
                        if (selp[i] >= 0) { run = true; break; }
                }
            }
            if (run)
                exl3_moe_cpu_forward_raw(
                    static_cast<int64_t>(job.layer),
                    reinterpret_cast<const at::Half*>(slot + off_x),
                    reinterpret_cast<const int32_t*>(slot + off_sel),
                    reinterpret_cast<const at::Half*>(slot + off_w),
                    reinterpret_cast<float*>(slot + off_out),
                    static_cast<int>(job.rows),
                    static_cast<int>(job.topk),
                    static_cast<int>(threads)
                );
        }

        store_release_u32(done + size_t(job.slot) * 16, job.seq);

        if (hprof)
        {
            const auto hp_t2 = std::chrono::steady_clock::now();
            auto ms = [](auto a, auto b)
                { return std::chrono::duration<double, std::milli>(b - a).count(); };
            const double g = ms(hp_prev_end, hp_t0), s = ms(hp_t0, hp_t1), c = ms(hp_t1, hp_t2);
            hp_gap += g; hp_spin += s; hp_comp += c;
            if (g > hp_gap_mx) hp_gap_mx = g;
            if (s > hp_spin_mx) hp_spin_mx = s;
            if (c > hp_comp_mx) hp_comp_mx = c;
            hp_prev_end = hp_t2;
            if (++hp_jobs % 64 == 0)
            {
                printf(" -- handoff prof (%ld jobs, ms/job avg|max): gap %.3f|%.3f "
                       "spin %.3f|%.3f compute %.3f|%.3f | empty %ld/64, "
                       "cpu-assign/row %.2f\n",
                       hp_jobs, hp_gap / 64, hp_gap_mx, hp_spin / 64, hp_spin_mx,
                       hp_comp / 64, hp_comp_mx, hp_empty,
                       hp_rows ? (double) hp_assign / (double) hp_rows : 0.0);
                fflush(stdout);
                hp_gap = hp_spin = hp_comp = 0.0;
                hp_gap_mx = hp_spin_mx = hp_comp_mx = 0.0;
                hp_empty = 0; hp_assign = 0; hp_rows = 0;
            }
        }
    }
    out:;
    stager.join();
}
