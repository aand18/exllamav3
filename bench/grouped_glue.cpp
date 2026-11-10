// Torch glue for the standalone grouped-MoE bench (grouped_kernel.cu).
// Compiled by MSVC only (no CUDA headers enter this TU beyond the stream
// type), so the CUDA-13.x CCCL / Windows-SDK clash that blocks a single-TU
// build on this box does not apply here. No custom autograd.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "grouped_kernel.h"

torch::Tensor grouped_moe(
    const torch::Tensor& x,
    const torch::Tensor& wg,
    const torch::Tensor& wu,
    const torch::Tensor& wd,
    const torch::Tensor& rw,
    const torch::Tensor& dist_exp,
    const torch::Tensor& seg_start,
    const torch::Tensor& seg_cnt,
    const torch::Tensor& row_perm,
    const torch::Tensor& slot_perm,
    const torch::Tensor& act,
    const torch::Tensor& out,
    int64_t topk)
{
    TORCH_CHECK(act.dtype() == at::kFloat && out.dtype() == at::kFloat,
                "grouped_moe: act/out scratch must be float32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int R2 = (int)act.size(0);
    const int I = (int)act.size(1);
    grouped_stageA(
        (const half*)x.data_ptr(), (const half*)wg.data_ptr(), (const half*)wu.data_ptr(),
        (const int32_t*)dist_exp.data_ptr(), (const int32_t*)seg_start.data_ptr(),
        (const int32_t*)seg_cnt.data_ptr(), (const int32_t*)row_perm.data_ptr(),
        (float*)act.data_ptr(), R2, (int)x.size(1), I, (int)dist_exp.numel(), stream);
    grouped_stageB(
        (const float*)act.data_ptr(), (const half*)wd.data_ptr(), (const half*)rw.data_ptr(),
        (const int32_t*)dist_exp.data_ptr(), (const int32_t*)seg_start.data_ptr(),
        (const int32_t*)seg_cnt.data_ptr(), (const int32_t*)row_perm.data_ptr(),
        (const int32_t*)slot_perm.data_ptr(), (float*)out.data_ptr(),
        I, (int)out.size(1), (int)topk, (int)dist_exp.numel(), stream);
    return out;
}

// No-op used by the driver to measure the pure pybind dispatch floor.
void noop() {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("grouped_moe", &grouped_moe, "grouped MoE (bench kernel)");
    m.def("noop", &noop, "dispatch floor probe");
}
