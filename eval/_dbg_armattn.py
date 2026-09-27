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
    # V-SOURCE TRIANGULATION on real records (no attention math):
    # torch-dequant vs image rows, triton-dequant vs image rows, over
    # SEALED groups only (unsealed groups serve from staging, not
    # records). fp16-image tolerance (~1e-3); looking for huge mismatch.
    from exllamav3.cache.kvarn import kvarn_wht_head
    with torch.inference_mode():
        Gs = torch.arange(lay0.records.shape[0], dtype=torch.int64,
                          device="cuda")
        bk_t, bv_t = lay0._dequant_groups_batched_torch(Gs)
        from exllamav3.modules.attention_fn.kvarn_triton import (
            kvarn_triton_dequant_groups)
        bk_x, bv_x = kvarn_triton_dequant_groups(
            lay0.records, lay0.layout, 4, 4, kvh, 2, do_wht=False)
        sg = lay0.sealed.nonzero().flatten()
        Ki = K[:8192].reshape(-1, 128, kvh, hd)[sg]
        Vi = V[:8192].reshape(-1, 128, kvh, hd)[sg]
        Kw = kvarn_wht_head(Ki.double(), hd).float()
        Vw = kvarn_wht_head(Vi.double(), hd).float()
        for tag, a, b in [("torchV-vs-imageV", bv_t[sg], Vw),
                          ("tritonV-vs-imageV", bv_x[sg], Vw),
                          ("torchK-vs-imageK", bk_t[sg], Kw),
                          ("tritonK-vs-imageK", bk_x[sg], Kw)]:
            dd = (a - b).abs()
            print(f"  {tag}: maxdiff={float(dd.max()):.3e} "
                  f"RMSE={float((dd ** 2).mean().sqrt()):.3e}", flush=True)
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
        print(f"  den scale: arm mean={float(den_bf.mean()):.3e} "
              f"ref mean={float(den_b_ref.mean()):.3e} "
              f"num scale: arm meanabs={float(num_b.abs().mean()):.3e} "
              f"ref meanabs={float(num_b_ref.abs().mean()):.3e}", flush=True)
        for hh in range(4):
            a = num_b[hh * 6:(hh + 1) * 6]
            b = num_b_ref[hh * 6:(hh + 1) * 6]
            dd = (a - b).abs()
            print(f"  kvhead {hh}: num maxdiff={float(dd.max()):.3e} "
                  f"RMSE={float((dd ** 2).mean().sqrt()):.3e}", flush=True)
        rep_rmse("num_b", num_b, num_b_ref)
        # ---- Entry-vs-dispatch isolation: production entry DIRECTLY on real
    # records (no dispatch torch code) vs torch ref AND vs arm output.
    # Matches ref => dispatch torch code guilty. Mismatches like arm =>
    # entry/kernel guilty on real-data mappings.
    with torch.inference_mode():
        import types as _t
        from exllamav3.modules.attention_fn.kvarn_triton import (
            kvarn_triton_online_serve, kvarn_triton_wht_rows as _w2)
        # LAST UNVERIFIED INPUT: dispatch's qw (production QWHT) vs the
        # direct call's qw_d (wht_rows). Every other input is provably
        # identical (same objects/constants). Compare first.
        layns = _t.SimpleNamespace(
            records=lay0.records, layout=lay0.layout, k_bits=4, v_bits=4,
            num_kv_heads=kvh, head_dim=hd, slices=lay0.slices)
        qw_d = _w2(Qf, hd)
        qw_a = lay0._ov_online_qw
        dq = (qw_a - qw_d).abs()
        print(f"  qw arm-vs-direct: maxdiff={float(dq.max()):.3e} "
              f"RMSE={float((dq ** 2).mean().sqrt()):.3e}", flush=True)
        Ew_d = _w2(lay0.exact_v.float(), hd)
        n_0d_d = torch.tensor([n2], dtype=torch.int32, device="cuda")
        out_d, flag_d = kvarn_triton_online_serve(
            layns, qw_d, Qf, lay0.exact_k, Ew_d, lay0._exact_rev,
            lay0.sealed, bt[0], n_0d_d, 6, 0.0625, 128, 128, 2,
            gc=65)
        dd = (out_d.float() - ref).abs()
        print(f"  direct-entry vs ref: maxdiff={float(dd.max()):.3e} "
              f"RMSE={float((dd ** 2).mean().sqrt()):.3e} flag={flag_d}",
              flush=True)
        dd2 = (out_d.float() - got).abs()
        print(f"  direct-entry vs arm: maxdiff={float(dd2.max()):.3e} "
              f"RMSE={float((dd2 ** 2).mean().sqrt()):.3e}", flush=True)
        # WHT-domain rows with kernel-consistent coverage. Online identity
        # (final acc = sum exp(s-m_final)*v) means single-pass torch refs
        # suffice; no iter replication needed. Guilty chunk types isolate
        # the path. Tolerance ~1e-2 (quant+fp assoc); bug is ~228 scale.
        Vw = kvarn_wht_head(V.double(), hd).float()
        mbuf = lay0._ov_serve_m[:, :6, :]
        lbuf = lay0._ov_serve_l[:, :6, :]
        abuf = lay0._ov_serve_acc[:, :6, :, :] \
            if hasattr(lay0, "_ov_serve_acc") else None
        nbad_m = nbad_l = nbad_a = 0
        shown = 0
        for c in range(65):
            rows = pos_all[c * 128:min((c + 1) * 128, n2)]
            cov = is_body[rows] | (is_st[rows] & okpos[rows])
            if not bool(cov.any()):
                continue
            Kb = K[rows][cov]
            Vb = Vw[rows][cov]
            for h in range(kvh):
                q = Qf[h * 6:(h + 1) * 6]
                s = (q @ Kb[:, h, :].T) * 0.0625
                m = s.amax(dim=-1)
                e = torch.exp(s - m.unsqueeze(-1))
                l = e.sum(dim=-1)
                a = e @ Vb[:, h, :]
                mk = mbuf[h, :, c]
                lk = lbuf[h, :, c]
                dm = float((mk - m).abs().max())
                dl = float(((lk - l).abs() / l.clamp_min(1e-6)).max())
                badm = dm > 5e-2
                badl = dl > 5e-2
                nbad_m += badm
                nbad_l += badl
                if abuf is not None:
                    ak = abuf[h, :, c, :]
                    da = float(((ak - a).abs() / a.abs().clamp_min(1e-3)).max())
                    bada = da > 5e-2
                    nbad_a += bada
                else:
                    da, bada = -1.0, False
                if (badm or badl or bada) and shown < 8:
                    print(f"  chunk {c} head {h}: dm={dm:.2e} "
                          f"dl_rel={dl:.2e} da_rel={da:.2e}", flush=True)
                    shown += 1
        print(f"  per-chunk: bad_m={nbad_m} bad_l={nbad_l} bad_a={nbad_a} "
              f"(of {65 * kvh} head-chunks)", flush=True)
    assert float((d ** 2).mean().sqrt()) < 5e-4
    print("ARMATTN PASS", flush=True)


if __name__ == "__main__":
    main()
