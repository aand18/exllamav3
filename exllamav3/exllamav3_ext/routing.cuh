#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>

// Deterministic int8 router projection (routing_gemm.cu)
bool routing_gemm_det_fits(const at::Tensor& hidden, const at::Tensor& gate_i8, const at::Tensor& gate_sb, const at::Tensor& scores);
void routing_gemm_det_(const at::Tensor& hidden, const at::Tensor& gate_i8, const at::Tensor& gate_sb, at::Tensor& scores, cudaStream_t stream);
void routing_gemm_det(const at::Tensor& hidden, const at::Tensor& gate_i8, const at::Tensor& gate_sb, at::Tensor scores);
void det_quant_weight(const at::Tensor& w, at::Tensor w_i8, at::Tensor w_s);
void det_math_test(const at::Tensor& x, at::Tensor y);

void routing_ds3_group
(
    const at::Tensor& scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor,
    const int act_fn,
    const int n_group,
    const int topk_group
);

void routing_gemv_f32
(
    const at::Tensor& hidden,
    const at::Tensor& gate_t,
    at::Tensor scores
);

void routing_ds3_nogroup
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor,
    const c10::optional<at::Tensor>& gate_t,
    const int act_fn
,
    const c10::optional<at::Tensor>& gate_i8,
    const c10::optional<at::Tensor>& gate_sb
);

void routing_ds3_nogroup_logits
(
    at::Tensor scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor,
    const bool use_topk,
    const int act_fn
);

void moe_split_map
(
    at::Tensor sel,
    const at::Tensor& map,
    at::Tensor hist,
    at::Tensor sel_cpu,
    const int64_t first_cpu_slot
);

void moe_split_issue
(
    at::Tensor sel,
    const c10::optional<at::Tensor>& map,
    const c10::optional<at::Tensor>& hist,
    const at::Tensor& y,
    const at::Tensor& w,
    int64_t h_sel_ptr,
    int64_t h_x_ptr,
    int64_t h_w_ptr,
    at::Tensor dev_count,
    const int64_t slot_idx,
    const int64_t hi,
    const int64_t first_cpu
);

void moe_split_collect_add
(
    at::Tensor final_out,
    int64_t h_out_ptr,
    const at::Tensor& dev_count,
    const int64_t slot_idx,
    const int64_t ho
);

// Per-expert collect (EXL3_MOE_PARTIAL_COMPLETION): fold one expert job's compact slice into the
// routed sum, straight from the pinned slot. The job's expert id is read from the descriptor word
// at h_expert_ptr (the GPU announced it), its slice offset and row count from
// h_segtab_ptr (2 * E_cpu int32, worker-written); n_cap is the slot's slice-row capacity
void moe_split_collect_add_expert
(
    at::Tensor final_out,
    int64_t h_out_ptr,
    int64_t h_rowidx_ptr,
    int64_t h_segtab_ptr,
    int64_t h_expert_ptr,
    const int64_t ho,
    const int64_t n_cap
);

void routing_sel_norm
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    const at::Tensor& selected,
    at::Tensor weights,
    const float scaling_factor,
    const c10::optional<at::Tensor>& gate_t,
    const int act_fn
,
    const c10::optional<at::Tensor>& gate_i8,
    const c10::optional<at::Tensor>& gate_sb
);

void routing_std
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const c10::optional<at::Tensor>& per_expert_scale,
    const c10::optional<at::Tensor>& gate_t,
    const c10::optional<at::Tensor>& bias
,
    const c10::optional<at::Tensor>& gate_i8,
    const c10::optional<at::Tensor>& gate_sb
);

void routing_std_logits
(
    at::Tensor scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const c10::optional<at::Tensor>& per_expert_scale,
    const bool use_topk
);
