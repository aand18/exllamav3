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
    k, v = lay0.get_kv(torch.tensor([n2], dtype=torch.int32, device="cuda"),
                       bt)
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
    assert float((d ** 2).mean().sqrt()) < 5e-4
    print("ARMATTN PASS", flush=True)


if __name__ == "__main__":
    main()
