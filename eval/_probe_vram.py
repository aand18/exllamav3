"""Cache-only VRAM accounting: per-tensor bytes per cache type.

Populates 8k ctx with fp16 and kvarn caches, then walks every layer
summing CUDA tensor bytes by attribute name. Shows where the bytes
go (records vs image vs pages). Untracked debug artifact.
"""
import sys

import torch

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_fp16, CacheLayer_kvarn
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.cache.kvarn import kvarn_parse_preset
from kvarn_microkld import SAMPLER_TEXT, populate, _bshape

MODEL = ("C:/Users/yoho/Downloads/tabbyAPI/models/"
         "Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3")
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
CHUNK = 4096
# Gospel presets: K4V4 safe baseline, K4V2 the pick (vLLM #46613).
PRESET = sys.argv[2] if len(sys.argv) > 2 else "kvarn4"


def account(cache, tag):
    by_name = {}
    total = 0
    nlayers = 0
    for lay in cache.layers.values():
        nlayers += 1
        seen = set()
        for name in dir(lay):
            if name.startswith("__"):
                continue
            try:
                t = getattr(lay, name)
            except Exception:
                continue
            if isinstance(t, torch.Tensor) and t.is_cuda and \
                    id(t) not in seen:
                seen.add(id(t))
                nbytes = t.nelement() * t.element_size()
                total += nbytes
                key = f"{type(lay).__name__}.{name}"
                by_name[key] = by_name.get(key, 0) + nbytes
    print(f"== {tag}: {nlayers} layers, total {total / 1e9:.2f}GB",
          flush=True)
    for k in sorted(by_name, key=by_name.get, reverse=True)[:14]:
        print(f"   {k}: {by_name[k] / 1e6:.1f}MB", flush=True)
    # Live residency (assigned slots) for windowed stores: steady-state
    # floor. Peak transient = floor + groups per evict interval (2).
    for lay in cache.layers.values():
        for attr, rev in (("stage", "_stage_rev"), ("exact", "_exact_rev")):
            tbl = getattr(lay, rev, None)
            if tbl is None:
                continue
            live = int((tbl >= 0).sum())
            key = f"{type(lay).__name__}.{attr}_live_slots"
            by_name[key] = max(by_name.get(key, 0), live)
    for k in sorted(by_name):
        if k.endswith("_live_slots"):
            print(f"   {k}: max {by_name[k]} live", flush=True)


def main():
    k_bits, v_bits = kvarn_parse_preset(PRESET)
    print(f"preset {PRESET} -> ({k_bits},{v_bits})", flush=True)
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    c_fp16 = Cache(model, max_num_tokens=NTOK + 256,
                   layer_type=CacheLayer_fp16)
    c_q8 = Cache(model, max_num_tokens=NTOK + 256,
                 layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
    c_kvarn = Cache(model, max_num_tokens=NTOK + 256,
                    layer_type=CacheLayer_kvarn,
                    k_bits=k_bits, v_bits=v_bits)
    model.load("cuda:0", progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    reps = max(16, (NTOK // 24) + 2)
    ids = tokenizer.encode(SAMPLER_TEXT * reps)[:, :NTOK]
    n = int(ids.shape[1])
    with torch.inference_mode():
        populate(model, c_fp16, ids, CHUNK, n)
        populate(model, c_q8, ids, CHUNK, n)
        populate(model, c_kvarn, ids, CHUNK, n)
    # Serve once so lazy temps (kvarn image) exist.
    se = torch.tensor([n], dtype=torch.int32, device="cuda:0")
    bt = torch.arange((NTOK + 256) // 256, dtype=torch.int32,
                      device="cuda:0").unsqueeze(0).expand(1, -1)
    with torch.inference_mode():
        for lay in c_fp16.layers.values():
            k, v = lay.get_kv(se, bt)
            del k, v
        for lay in c_q8.layers.values():
            k, v = lay.get_kv(se, bt)
            del k, v
        for lay in c_kvarn.layers.values():
            k, v = lay.get_kv(se, bt)
            del k, v
    torch.cuda.synchronize()
    # Fire one imageless arm decode so the online serve temps exist
    # before accounting (they are real serving VRAM: serve partials +
    # tail temps, per layer). Fail-soft: without TRITON/IMAGELESS the
    # arm declines and there is nothing extra to count. The re-stored
    # last row is idempotent for accounting (same slots, same shapes).
    import os as _os
    if _os.environ.get("EXL3_KVARN_IMAGELESS") == "1":
        try:
            p = {"cache": c_kvarn, "attn_mode": "flash_attn",
                 "batch_shape": (1, _bshape(ids.shape[1])), "past_len": n - 1}
            out = model.forward(ids[:, n - 1:n], p)
            del out
            torch.cuda.synchronize()
            print("arm fired once for accounting", flush=True)
        except Exception as e:
            print(f"arm fire skipped: {type(e).__name__}: {e}", flush=True)
    account(c_fp16, "fp16 cache-only")
    account(c_q8, "q8 cache-only")
    account(c_kvarn, f"kvarn({PRESET}) cache-only")


if __name__ == "__main__":
    main()
