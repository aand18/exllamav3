"""Isolate per-layer steady-decode costs with CUDA events (proper sync).

Populates 8192 ctx on ONE kvarn layer, then mimics decode (advancing
positions, production get_kv -> update_kv order, no model):
  pair  : get_kv(se) + update_kv(se, img, 1)   [production per-step mix]
  serve : get_kv(se) + unoverlay               [serve half of the pair]
  touch : _touch_batch(se, bt, 1)              [stateless, repeatable]
Derives store+evict ~= pair - serve - touch.
Reports ms/call (synchronized). Needs GPU box. Untracked debug artifact.
"""
import torch

from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.cache import CacheLayer_kvarn
from exllamav3.cache.kvarn import kvarn_parse_preset
from exllamav3.modules.attention_fn.kvarn_triton import kvarn_triton_unoverlay
from kvarn_microkld import SAMPLER_TEXT, populate

MODEL = ("C:/Users/yoho/Downloads/tabbyAPI/models/"
         "Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3")
NTOK, CHUNK, ITERS = 8192, 4096, 200


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
    max_tok = NTOK + 512
    bt = torch.arange(max_tok // 256, dtype=torch.int32,
                      device="cuda:0").unsqueeze(0).expand(1, -1).contiguous()
    st = {"i": 0}

    def se():
        return torch.tensor([n + st["i"]], dtype=torch.int32, device="cuda:0")

    with torch.inference_mode():
        # Device-side high-water marks (no DtoH sync per step: maximum is
        # reduced on-GPU, read once at the end; timing loops unaffected).
        hw_exact = torch.zeros((), dtype=torch.int64, device="cuda:0")
        hw_stage = torch.zeros((), dtype=torch.int64, device="cuda:0")

        def track():
            hw_exact.copy_(torch.maximum(
                hw_exact, (lay0._exact_rev >= 0).sum()))
            hw_stage.copy_(torch.maximum(
                hw_stage, (lay0._stage_rev >= 0).sum()))

        # Warm the image + overlay machinery into steady state.
        for _ in range(10):
            k, v = lay0.get_kv(se(), bt)
            del k, v
            lay0.update_kv(se(), bt, lay0._img_k, lay0._img_v, 1)
            st["i"] += 1

        st["i"] = 0

        def do_pair():
            s = se()
            k, v = lay0.get_kv(s, bt)
            del k, v
            lay0.update_kv(s, bt, lay0._img_k, lay0._img_v, 1)
            track()
            st["i"] += 1
        # Pair consumes positions; run it last.
        t_touch = hot(lambda: lay0._touch_batch(se(), bt, 1))

        st["i"] = 0

        def do_serve():
            k, v = lay0.get_kv(se(), bt)
            del k, v
            kvarn_triton_unoverlay(lay0._img_k, lay0._img_v, lay0)
            track()
            st["i"] += 1
        t_serve = hot(do_serve)

        st["i"] = 0
        t_pair = hot(do_pair)

    t_store = t_pair - t_serve - t_touch
    print(f"pair per layer:  {t_pair:.3f} ms/call", flush=True)
    print(f"serve per layer: {t_serve:.3f} ms/call", flush=True)
    print(f"touch per layer: {t_touch:.3f} ms/call", flush=True)
    print(f"store+evict per layer ~= {t_store:.3f} ms/call", flush=True)
    print(f"pair x16 layers: {t_pair * 16:.2f} ms/step", flush=True)
    torch.cuda.synchronize()
    print(f"high-water exact slots (lay0, decode): {int(hw_exact)}",
          flush=True)
    print(f"high-water stage slots (lay0, decode): {int(hw_stage)}",
          flush=True)


if __name__ == "__main__":
    main()
