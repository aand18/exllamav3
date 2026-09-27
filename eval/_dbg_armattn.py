"""Direct end-to-end validation of the imageless dispatch arm (needs GPU).

KLD continuation scoring never exercises the arm (q_len>>1 declines to
get_kv), and -dec timing doesn't check values -- so the arm's output
correctness was UNVERIFIED until this script. It calls the arm on a real
populated layer, then compares against torch SDPA over get_kv full rows
(flag toggled mid-process: OFF = proven image path reference).

Usage: python eval/_dbg_armattn.py
Gate: RMSE < 5e-4 (fp16-rounding envelope), flag clean, arm fired.
"""
import os

import torch

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_kvarn
from exllamav3.cache.kvarn import kvarn_parse_preset
from kvarn_microkld import SAMPLER_TEXT, populate

MODEL = ("C:/Users/yoho/Downloads/tabbyAPI/models/"
         "Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3")
NTOK, CHUNK = 8192, 4096


def main():
    assert os.environ.get("EXL3_KVARN_IMAGELESS") == "1"
    assert os.environ.get("EXL3_KVARN_TRITON") == "1"
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
    kvh, hd = lay0.num_kv_heads, lay0.head_dim
    qh = 24
    assert kvh * 6 == qh
    max_tok = NTOK + 512
    bt = torch.arange(max_tok // 256, dtype=torch.int32,
                      device="cuda").unsqueeze(0).expand(1, -1).contiguous()
    torch.manual_seed(7)
    Q = torch.randn(1, 1, qh, hd, dtype=torch.float16, device="cuda")
    K1 = torch.randn(1, 1, kvh, hd, dtype=torch.float16, device="cuda")
    V1 = torch.randn_like(K1)
    seqlens = torch.tensor([n], dtype=torch.int32, device="cuda")
    from exllamav3.modules.attention_fn import dispatch as D
    out = D._try_kvarn_online_decode(
        Q, K1, V1, lay0, 0, 0, bt, seqlens, 1, 0.0625, True, None,
        0.0, None, None, None)
    assert out is not None, "arm declined (gate?)"
    print("arm fired, out shape:", tuple(out.shape), flush=True)
    # Reference at n+1 (arm stored K1/V1 first): flag OFF = proven image.
    os.environ["EXL3_KVARN_IMAGELESS"] = "0"
    n2 = n + 1
    with torch.inference_mode():
        k, v = lay0.get_kv(torch.tensor([n2], dtype=torch.int32,
                                        device="cuda"), bt)
    K = k[bt[0]].reshape(-1, kvh, hd)[:n2].float()
    V = v[bt[0]].reshape(-1, kvh, hd)[:n2].float()
    Qf = Q[0, 0].float()
    outs = []
    for h in range(kvh):
        q = Qf[h * 6:(h + 1) * 6]
        p = torch.softmax((q @ K[:, h, :].T) * 0.0625, dim=-1)
        outs.append(p @ V[:, h, :])
    ref = torch.cat(outs)
    got = out[0, 0].float()
    d = (got - ref).abs()
    print(f"arm maxdiff={float(d.max()):.3e} "
          f"RMSE={float((d ** 2).mean().sqrt()):.3e}", flush=True)

    # ---- Stage decomposition: which stage diverges? ----
    # True-row refs over partitions; arm internals recomputed like dispatch.
    from exllamav3.cache.kvarn import KVAR_N_SINK_TOKENS
    from exllamav3.cache.kvarn import kvarn_wht_head
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_wht_rows)
    gps = PAGE_SIZE // 128
    teff = int(lay0.tail_effective)
    sn = KVAR_N_SINK_TOKENS if lay0.has_sink else 0
    pos_all = torch.arange(n2, device="cuda")
    gg = bt[0][pos_all // PAGE_SIZE] * gps + (pos_all % PAGE_SIZE) // 128
    okpos = (lay0._exact_rev[gg] >= 0)
    is_body = (pos_all >= sn) & (pos_all < n2 - teff)
    is_st = ~is_body

    def part_stats(Kp, Vp, tag):
        """torch online stats over a row subset; (num, den, m) or Nones."""
        if Kp.shape[0] == 0:
            print(f"  {tag}: EMPTY subset", flush=True)
            return None, None, None
        ns, ds, ms = [], [], []
        for h in range(kvh):
            q = Qf[h * 6:(h + 1) * 6]
            s = (q @ Kp[:, h, :].T) * 0.0625
            m = s.amax(dim=-1)
            e = torch.exp(s - m.unsqueeze(-1))
            den = e.sum(dim=-1)
            num = e @ Vp[:, h, :]
            ns.append(num)
            ds.append(den)
            ms.append(m)
        return torch.cat(ns), torch.cat(ds), torch.cat(ms)

    def rep_rmse(tag, a, b):
        if a is None or b is None:
            return
        dd = (a - b).abs()
        print(f"  {tag}: maxdiff={float(dd.max()):.3e} "
              f"RMSE={float((dd ** 2).mean().sqrt()):.3e}", flush=True)

    with torch.inference_mode():
        num_b_ref, den_b_ref, m_b_ref = part_stats(
            K[is_body | (is_st & okpos)], V[is_body | (is_st & okpos)],
            "ref_body+atail")
        num_u_ref, den_u_ref, m_u_ref = part_stats(
            K[is_st & ~okpos], V[is_st & ~okpos], "ref_utail")
        num_bo_ref, den_bo_ref, m_bo_ref = part_stats(
            K[is_body], V[is_body], "ref_bodyonly")
        # Arm internals, recomputed exactly like dispatch.
        mb = lay0._ov_serve_m[:, :6, :]
        lb = lay0._ov_serve_l[:, :6, :]
        m_bf = mb.amax(dim=2).reshape(qh)
        den_bf = (lb * torch.exp(mb - mb.amax(dim=2).unsqueeze(-1))) \
            .sum(dim=2).reshape(qh)
        out_b = lay0._ov_serve_out
        print(f"  out_b: isnan={int(torch.isnan(out_b).sum())} "
              f"isinf={int(torch.isinf(out_b).sum())} "
              f"maxabs={float(out_b.abs().max()):.3e}", flush=True)
        # PRIME SUSPECT CHECK: triton WHT vs torch head-WHT (must be
        # bit-exact per its docstring; a broken WHT explains correct
        # out_b + wrong num_b with near-right den_b).
        w_ref = kvarn_wht_head(out_b.double(), hd).float()
        w_got = kvarn_triton_wht_rows(out_b, hd)
        wd = (w_got - w_ref).abs()
        print(f"  wht_rows: maxdiff={float(wd.max()):.3e} "
              f"RMSE={float((wd ** 2).mean().sqrt()):.3e}", flush=True)
        num_b = w_got * den_bf.reshape(qh, 1)
        rep_rmse("m_b", m_bf, m_b_ref)
        rep_rmse("den_b", den_bf, den_b_ref)
        rep_rmse("num_b", num_b, num_b_ref)
        rep_rmse("num_b-vs-bodyonly", num_b, num_bo_ref)
        # Arm torch-tail recomputed like dispatch (bmm + exrev mask).
        Kt, Vt = lay0.kvarn_online_tail(n2, bt[0])
        tp = torch.cat([torch.arange(sn, device="cuda"),
                        torch.arange(max(0, n2 - teff), n2,
                                     device="cuda")]).long()
        tg = bt[0][tp // PAGE_SIZE] * gps + (tp % PAGE_SIZE) // 128
        ok = (lay0._exact_rev[tg] < 0).to(torch.float32)
        Qh = Qf.reshape(kvh, 6, hd)
        st = torch.bmm(Qh, Kt.permute(1, 2, 0)) * 0.0625
        tm = st.amax(dim=-1)
        pe = torch.exp(st - tm.unsqueeze(-1)) * ok
        rep_rmse("tail_m", tm.reshape(qh), m_u_ref)
        rep_rmse("tail_den", pe.sum(dim=-1).reshape(qh), den_u_ref)
        rep_rmse("tail_num",
                 torch.bmm(pe, Vt.permute(1, 0, 2)).reshape(qh, hd),
                 num_u_ref)
    assert float((d ** 2).mean().sqrt()) < 5e-4
    print("ARMATTN PASS", flush=True)


if __name__ == "__main__":
    main()
