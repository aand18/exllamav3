"""Piece-timing for the imageless dispatch arm, per stage (needs GPU).

Populates 8k ctx, then mimics single-row decode on lay0 with production
calls in dispatch order, timing each stage with CUDA events:
  store  : update_kv_direct (1 row)
  qwht   : kvarn_triton_qwht
  eref   : exact_v.float + wht_rows (full refresh, as dispatch does now)
  serve  : kvarn_triton_online_serve (kernel+combine)
  stats  : per-head (m, den) + body unwrap
  tail   : kvarn_online_tail + masked per-head torch block
  merge  : orig-domain online merge
  mask   : tail-position assignment mask build
  armsum : pieces summed (sanity vs full below)
  full   : _try_kvarn_online_decode end-to-end (ground truth per layer)
Usage: python eval/_probe_arm.py
"""
import torch

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_kvarn
from exllamav3.cache.kvarn import (
    kvarn_parse_preset, KVAR_N_SINK_TOKENS, KVAR_N_GROUP)
from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attention_fn.kvarn_triton import (
    kvarn_triton_qwht, kvarn_triton_wht_rows, kvarn_triton_online_serve,
    _kvarn_online_buffers)
from kvarn_microkld import SAMPLER_TEXT, populate

MODEL = ("C:/Users/yoho/Downloads/tabbyAPI/models/"
         "Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3")
NTOK, CHUNK, ITERS = 8192, 4096, 100


def hot(fn, iters=ITERS):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def main():
    k_bits, v_bits = kvarn_parse_preset("kvarn4")
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=NTOK + 512,
                  layer_type=CacheLayer_kvarn, k_bits=k_bits, v_bits=v_bits)
    model.load("cuda:0", progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    reps = max(16, (NTOK // 24) + 2)
    ids = tokenizer.encode(SAMPLER_TEXT * reps)[:, :NTOK]
    n = int(ids.shape[1])
    states, _ = populate(model, cache, ids, CHUNK, n)
    del states
    torch.cuda.synchronize()
    lay0 = next(iter(cache.layers.values()))
    kvh, hd, sl = lay0.num_kv_heads, lay0.head_dim, lay0.slices
    qh, qpk = 24, 6
    assert kvh * qpk == qh and sl == 2
    dev = "cuda"
    torch.manual_seed(11)
    Q1 = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    K1 = torch.randn(1, 1, kvh, hd, dtype=torch.float16, device="cuda")
    V1 = torch.randn_like(K1)
    q = Q1.reshape(1, 1, qh, hd)
    seqlens = torch.tensor([n], dtype=torch.int32, device="cuda")
    max_tok = NTOK + 512
    bt = torch.arange(max_tok // 256, dtype=torch.int32,
                      device="cuda").unsqueeze(0).expand(1, -1).contiguous()
    gps = PAGE_SIZE // KVAR_N_GROUP
    scale, sink_n, tail_eff = 0.0625, 128, int(lay0.tail_effective)
    qpad = 1 << (qpk - 1).bit_length()
    sscale = 0.7071067811865475
    gc_eff = (n + 127) // 128

    from exllamav3.modules.attention_fn import dispatch as D
    t_store = hot(lambda: lay0.update_kv_direct(seqlens, bt, K1, V1, 1))
    _mb, _lb, _ab, qw, qs, _o = _kvarn_online_buffers(
        lay0, qh, qpad, hd, torch.device("cuda"))
    t_qwht = hot(lambda: kvarn_triton_qwht(Q1, qs, qw, sl, sscale),
                 iters=500)
    t_eref = hot(lambda: kvarn_triton_wht_rows(lay0.exact_v.float(), hd),
                 iters=200)
    Qf = Q1.float()
    Ew = kvarn_triton_wht_rows(lay0.exact_v.float(), hd)
    n_0d = seqlens[:1] + 1

    def do_serve():
        return kvarn_triton_online_serve(
            lay0, qw, Qf, lay0.exact_k, Ew, lay0._exact_rev, lay0.sealed,
            bt[0], n_0d, qpk, scale, sink_n, tail_eff, gps, gc=gc_eff)
    o_b, f_b = do_serve()
    t_serve = hot(lambda: do_serve()[0])

    def do_stats():
        mb = lay0._ov_serve_m[:, :qpk, :]
        lb = lay0._ov_serve_l[:, :qpk, :]
        m_b = mb.amax(dim=2)
        den_b = (lb * torch.exp(mb - m_b.unsqueeze(-1))).sum(dim=2)
        num_b = kvarn_triton_wht_rows(o_b, hd) * den_b.reshape(qh, 1)
        return m_b.reshape(qh), den_b.reshape(qh), num_b
    m_b, den_b, num_b = do_stats()
    t_stats = hot(do_stats, iters=500)

    sn_ = min(128, n)
    t0_ = max(0, n - tail_eff)
    tpos = torch.cat([torch.arange(sn_, device=dev),
                      torch.arange(t0_, n, device=dev)]).long()

    def do_mask():
        tg = bt[0][tpos // PAGE_SIZE] * gps + (tpos % PAGE_SIZE) // KVAR_N_GROUP
        return (lay0._exact_rev[tg] >= 0).to(torch.float32)
    ok = do_mask()
    t_mask = hot(do_mask, iters=500)

    def do_tail():
        Kt, Vt = lay0.kvarn_online_tail(n, bt[0])
        t_ms, t_ns, t_ds = [], [], []
        for h in range(kvh):
            qh_ = Q1[h * qpk:(h + 1) * qpk].float()
            st = (qh_ @ Kt[:, h, :].T) * scale
            tm = st.amax(dim=-1)
            pe = torch.exp(st - tm.unsqueeze(-1)) * ok
            t_ms.append(tm)
            t_ns.append(pe @ Vt[:, h, :])
            t_ds.append(pe.sum(dim=-1))
        return torch.cat(t_ms), torch.cat(t_ns), torch.cat(t_ds)
    tail_m, tail_num, tail_den = do_tail()
    t_tail = hot(do_tail, iters=200)

    def do_merge():
        m_g = torch.maximum(m_b, tail_m)
        eb = torch.exp(m_b - m_g)
        et = torch.exp(tail_m - m_g)
        den = den_b * eb + tail_den * et
        num = num_b * eb.unsqueeze(-1) + tail_num * et.unsqueeze(-1)
        return (num / den.unsqueeze(-1)).half()
    do_merge()
    t_merge = hot(do_merge, iters=500)

    def do_full():
        return D._try_kvarn_online_decode(
            q, K1, V1, lay0, 0, 0, bt, seqlens, 1, scale, True, None,
            0.0, None, None, None)
    r = do_full()
    print("full arm fired:", r is not None, flush=True)
    t_full = hot(do_full, iters=50)
    parts = [("store", t_store), ("qwht", t_qwht), ("eref", t_eref),
             ("serve", t_serve), ("stats", t_stats), ("mask", t_mask),
             ("tail", t_tail), ("merge", t_merge)]
    for name, t in parts:
        print(f"{name:6s} per layer: {t * 1e3:7.1f} us", flush=True)
    print(f"sum   per layer: {sum(t for _, t in parts) * 1e3:7.1f} us",
          flush=True)
    print(f"full  per layer: {t_full * 1e3:7.1f} us", flush=True)


if __name__ == "__main__":
    main()
