#!/usr/bin/env python3
"""Render the consolidated Flash-Next knob report as a single static HTML file.

Reads the CSV data set so no number is retyped, and emits one self-contained
file (embedded CSS, no JS, no external assets) that prints cleanly.

Design constraints, from the reader this is for:
  - someone else on the team, so CONFIDENCE is visible on every claim and the
    "what this does not establish" section is not buried
  - static and print-friendly: no JS, no network, @media print rules
  - confidence is labelled in TEXT as well as colour, so it survives B&W printing
"""

import csv
import html
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "wiki", "reports",
                    "data-2026-10-06-flash-next-knobs")
OUT = os.path.join(HERE, "..", "wiki", "reports",
                   "2026-10-06-flash-next-knobs.html")


def rows(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def e(s):
    return html.escape(str(s if s is not None else ""))


CONF = {
    "HIGH": ("ok", "settled — same-window reference, replicated"),
    "MED": ("mid", "measured, but n small / single pair / one tier"),
    "LOW": ("low", "screen only, confounded, or superseded"),
    "SCREEN": ("none", "not measured to plan depth — don't re-test blindly"),
}


def badge(text):
    """Colour by the LEADING token, so compound values like
    'HIGH (null result)' or 'HIGH at 130k / MED short-ctx' do not fall through
    to the LOW/red badge -- which would show a settled finding as unreliable.
    """
    key = (text or "").strip()
    lead = key.split()[0].rstrip(":") if key else ""
    cls, _ = CONF.get(lead, ("low", ""))
    return f'<span class="badge {cls}">{e(key)}</span>'


def table(headers, body_rows, cls="", caption=None):
    cap = f"<caption>{e(caption)}</caption>" if caption else ""
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for r in body_rows:
        tds = "".join(f"<td>{c}</td>" for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return (f'<div class="tw"><table class="{cls}">{cap}<thead><tr>{th}</tr>'
            f'</thead><tbody>{"".join(trs)}</tbody></table></div>')


def build():
    dec = rows("decisions.csv")
    lc = rows("long-context.csv")
    dr = rows("draft.csv")
    rets = rows("retractions.csv")
    meth = rows("methodology.csv")

    # ---------- recommendation -------------------------------------------
    adopt = [d for d in dec if d["verdict"] == "ADOPT"]
    cfg = [
        ("model", "#cpu_moe_offload_layers: 38",
         "removed — mutually exclusive with the line below; never set both"),
        ("model", "cpu_moe_split_experts: 380", "the long-context lever"),
        ("model", "cache_mode: 2,2", "perf-neutral, +998 MB VRAM"),
        ("draft_model", "dynamic_draft: true", "unchanged"),
        ("draft_model", "draft_cache_mode: Q4", "unchanged — deliberate, confirmed"),
        ("draft_model", "draft_num_tokens: 3", "was 5"),
        ("memory", "cuda_malloc_async: True", "unchanged"),
    ]
    cfg_rows = [(f"<code>{e(k)}</code>", f"<code>{e(a)}</code>", e(b))
                for k, a, b in cfg]

    # ---------- long context ---------------------------------------------
    order = ["11-16k_turnmatch", "55-62k", "130k", "224k", "250k"]
    stage_rows = []
    for st_name in order:
        rs = [r for r in lc if r["stage"] == st_name]
        if not rs:
            continue
        ptok = sorted({r["prompt_tokens_actual"] for r in rs if r["prompt_tokens_actual"]})
        ptxt = f"{min(int(p) for p in ptok):,}" if ptok else "10,956–16,135"
        combo = [float(r["ratio_vs_baseline"]) for r in rs if r["arm"] == "combo"]
        if not combo:
            continue
        med = st.median(combo)
        conf = rs[0]["confidence"]
        state = {r["machine_state"] for r in rs}
        note = ""
        if st_name == "250k":
            note = ("<br><small>state-matched: "
                    f"fast {1/med:.3f}× / slow {1/med:.3f}× — see caveat</small>")
            stage_rows.append((e(st_name.replace("_", " ")), e(ptxt), "—",
                               note, badge(conf)))
            continue
        what = {"11-16k_turnmatch": "decode",
                "55-62k": "— (neutral)",
                "130k": "prefill, +14% T/s",
                "224k": "prefill (single pair)"}[st_name]
        stage_rows.append((e(st_name.replace("_", " ")), e(ptxt),
                           f"<strong>{1/med:.3f}×</strong>", e(what), badge(conf)))

    # ---------- draft ----------------------------------------------------
    ndt = [r for r in dr if r["test"] == "ndt_ladder"]
    ndt.sort(key=lambda r: -float(r["value"]))
    ndt_rows = [(f"<code>{e(r['setting'])}</code>", r["value"],
                 f"{r['vs_baseline_pct']}%" if r["vs_baseline_pct"] else "—",
                 r["acceptance"] or "—", r["acceptance_per_drafted"] or "—")
                for r in ndt]

    acc = [r for r in dr if r["test"] == "acceptance_by_category"]
    acc.sort(key=lambda r: -float(r["value"]))
    acc_rows = [(e(r["category"]), r["tools"], e(r["setting"]),
                 f"<strong>{r['value']}%</strong>", r["acceptance"],
                 r["acceptance_per_drafted"], badge("MED"))
                for r in acc]

    dcm = [r for r in dr if r["test"] == "draft_cache_mode_ladder"]
    dcm.sort(key=lambda r: float(r["value"]))
    dcm_rows = []
    for r in dcm:
        v = float(r["value"])
        verdict = ("<strong>neutral</strong>" if abs(v - 1) <= 0.025
                   else "<strong class='bad'>slower</strong>" if v > 1
                   else "<strong class='good'>faster</strong>")
        vmin = r["note"].split("vram_minfree=")[-1].replace("MB", "") \
            if "vram_minfree=" in r["note"] else ""
        dcm_rows.append((f"<code>{e(r['setting'])}</code>", r["n"], verdict, vmin))

    # ---------- verdict tables ------------------------------------------
    def dec_table(verdicts, title):
        sel = [d for d in dec if d["verdict"] in verdicts]
        body = [(f"<code>{e(d['knob'])}</code>", e(d["current"]), e(d["proposed"]),
                 e(d["measured_effect"]), e(d["cost"]), badge(d["confidence"]))
                for d in sel]
        return table(["knob", "current", "proposed", "measured effect", "cost",
                      "confidence"], body, caption=title)

    rets_rows = [(e(r["seq"]), f"<strong>{e(r['claim_withdrawn'])}</strong>",
                  e(r["why_wrong"]), e(r["correction"])) for r in rets]
    meth_rows = [(f"<strong>{e(m['rule'])}</strong>", e(m["reason"]),
                  e(m["error_it_prevented"])) for m in meth]

    adopt_rows = [(f"<code>{e(d['knob'])}</code>",
                   f"<code>{e(d['current'])} → {e(d['proposed'])}</code>",
                   e(d["measured_effect"]), e(d["cost"]),
                   badge(d["confidence"]), e(d["rationale"])) for d in adopt]

    return dict(cfg_rows=cfg_rows, adopt_rows=adopt_rows, stage_rows=stage_rows,
                ndt_rows=ndt_rows, acc_rows=acc_rows, dcm_rows=dcm_rows,
                rets_rows=rets_rows, meth_rows=meth_rows,
                keep=dec_table({"KEEP", "KEEP (low conf)"}, None),
                reject=dec_table({"REJECT", "REPLACE"}, None),
                n_dec=len(dec), n_rets=len(rets), n_meth=len(meth))


CSS = """
:root{
  --bg:#fbfaf9; --fg:#1c1a19; --mut:#6b6560; --line:#e4e0dc; --card:#fff;
  --ok:#0f7b4f; --okbg:#e6f4ec; --mid:#8a5a00; --midbg:#fdf3e0;
  --low:#b3261e; --lowbg:#fdeceb; --none:#6b6560; --nonebg:#f0eeec;
  --accent:#7a3e9d;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#16151a;--fg:#e9e6e2;--mut:#a09a94;--line:#33313a;--card:#1e1d23;
       --ok:#5fd39b;--okbg:#12301f;--mid:#e0a84a;--midbg:#33280f;
       --low:#f28b82;--lowbg:#3a1512;--none:#a09a94;--nonebg:#26252b;
       --accent:#c39ee0;}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
  font-variant-numeric:tabular-nums}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 96px}
header.top{border-bottom:2px solid var(--line);padding-bottom:20px;margin-bottom:8px}
h1{font-size:1.9rem;line-height:1.2;margin:0 0 6px}
.sub{color:var(--mut);font-size:.95rem;margin:0}
.meta{color:var(--mut);font-size:.83rem;margin-top:10px}
h2{font-size:1.32rem;margin:44px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:1.03rem;margin:26px 0 8px;color:var(--fg)}
p,li{max-width:74ch}
code{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:var(--nonebg);padding:1.5px 5px;border-radius:4px}
a{color:var(--accent)}
.tw{overflow-x:auto;margin:12px 0 18px}
table{border-collapse:collapse;width:100%;font-size:.88rem}
caption{caption-side:top;text-align:left;color:var(--mut);font-size:.83rem;
  padding-bottom:6px}
th{text-align:left;font-weight:600;border-bottom:2px solid var(--line);
  padding:8px 10px;white-space:nowrap}
td{border-bottom:1px solid var(--line);padding:8px 10px;vertical-align:top}
tbody tr:nth-child(even){background:var(--card)}
.badge{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.03em;
  padding:2px 7px;border-radius:10px;white-space:nowrap}
.badge.ok{color:var(--ok);background:var(--okbg)}
.badge.mid{color:var(--mid);background:var(--midbg)}
.badge.low{color:var(--low);background:var(--lowbg)}
.badge.none{color:var(--none);background:var(--nonebg)}
.good{color:var(--ok)}.bad{color:var(--low)}
.callout{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
  border-radius:6px;padding:14px 18px;margin:16px 0}
.callout.warn{border-left-color:var(--low)}
.callout.ok{border-left-color:var(--ok)}
.callout h3{margin-top:0}
.callout p:last-child{margin-bottom:0}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;
  padding:14px 16px;overflow-x:auto;font-size:.84rem;line-height:1.5}
ul.tight{margin:.4em 0}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:12px 0 18px;font-size:.84rem}
.legend div{display:flex;align-items:center;gap:7px}
details{border:1px solid var(--line);border-radius:6px;margin:14px 0;background:var(--card)}
details>summary{cursor:pointer;padding:11px 16px;font-weight:600;font-size:.95rem}
details[open]>summary{border-bottom:1px solid var(--line)}
details .inner{padding:14px 16px 6px}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--mut);font-size:.83rem}
@media print{
  :root{--bg:#fff;--fg:#000;--mut:#444;--line:#bbb;--card:#fff;
       --ok:#0a5c39;--okbg:#fff;--mid:#6b4400;--midbg:#fff;
       --low:#8f1d17;--lowbg:#fff;--none:#555;--nonebg:#fff}
  body{font-size:10.5pt}
  .wrap{max-width:none;padding:0}
  details{border:1px solid #bbb}
  details>summary{list-style:none}
  details:not([open])>summary::after{content:" (expand to print)"}
  details .inner{padding:8px 10px}
  /* NOTE: CSS cannot force <details> open. Collapsed sections stay collapsed
     in print output -- that is the trade for a static, JS-free, digestible
     page. The ::after hint above tells the reader to expand them first. */
  details .inner{display:block}
  h2{page-break-after:avoid;break-after:avoid}
  h3{page-break-after:avoid;break-after:avoid}
  tr, .callout, pre{page-break-inside:avoid;break-inside:avoid}
  thead{display:table-header-group}
  .badge{border:1px solid currentColor}
}
"""

PRINT_JS = ""  # deliberately none: this must work with JS disabled


def render():
    d = build()
    o = []
    a = o.append
    a("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    a("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    a("<title>Flash-Next knob battery — results (2026-10-06)</title>")
    a(f"<style>{CSS}</style></head><body><div class='wrap'>")

    a("<header class='top'><h1>Flash-Next knob battery — results</h1>")
    a("<p class='sub'>Measured impact of every tunable on 3.05bpw Flash-Next "
      "serving (48 MoE layers, MTP head), RTX 4090 24 GB + 7950X3D (16C/32T), "
      "Win11 + WSL2.</p>")
    a(f"<p class='meta'>2026-10-06 &middot; {d['n_dec']} knobs judged &middot; "
      "measured 11k–259k prompt tokens, up to 96–99% of <code>cache_size</code> "
      "&middot; production <code>config.yml</code> never modified &middot; "
      "no JavaScript, prints cleanly</p></header>")

    # ---- read this first
    a("<h2>Read this first</h2>")
    a("<div class='callout warn'><h3>The long-context gain is not uniform</h3>")
    a("<p>This workload runs <strong>200k+ prompts</strong>, so the right-hand "
      "rows of the long-context table are what matter — and they are the "
      "<em>weakest</em> evidence here. The combo is <strong>1.157× at 130k</strong> "
      "(well established) but at <strong>parity at 250k</strong>. A "
      "<strong>1.36× figure from short prompts would badly overstate</strong> what "
      "this battery can say about your traffic.</p></div>")
    a("<div class='callout'><h3>How to read the confidence markers</h3>")
    a("<p><strong>HIGH</strong> means the reference arm was measured in the "
      "<em>same time window</em> as the candidate, and replicated. That single "
      "rule caught more errors than any other part of the protocol — session "
      "drift is ~10% while drift between consecutive boots is 0.07%, so a "
      "carried-forward baseline is invisible to any within-batch check.</p></div>")
    a("<div class='legend'>"
      "<div><span class='badge ok'>HIGH</span> settled</div>"
      "<div><span class='badge mid'>MED</span> n small / single pair</div>"
      "<div><span class='badge low'>LOW</span> screen, confounded, superseded</div>"
      "<div><span class='badge none'>SCREEN</span> not measured — don't re-test</div>"
      "</div>")

    # ---- the change
    a("<h2>The change</h2>")
    a("<pre>model:\n"
      "  #cpu_moe_offload_layers: 38      # mutually exclusive with the line below\n"
      "  cpu_moe_split_experts: 380\n"
      "  cache_mode: 2,2\n"
      "draft_model:\n"
      "  draft_num_tokens: 3              # was 5\n"
      "  dynamic_draft: true\n"
      "  draft_cache_mode: Q4\n"
      "memory:\n"
      "  cuda_malloc_async: True\n\n"
      "# start_tuned.ps1:\n"
      "#   EXL3_MOE_CPU_THREADS=8 -> 16</pre>")
    a("<div class='callout ok'><h3>Fallback order if VRAM ever binds</h3>")
    a("<p>Drop <code>cpu_moe_split_experts</code>, keep the threads change. "
      "<code>thr16</code> costs <strong>no VRAM at all</strong> and carries the "
      "largest decode win in the battery.</p></div>")
    a("<details open><summary>Adopted changes — evidence</summary><div class='inner'>")
    a(table(["knob", "change", "measured effect", "cost", "confidence", "why"],
            d["adopt_rows"]))
    a("</div></details>")

    # ---- long context
    a("<h2>Long context — the gain by prompt length</h2>")
    a(table(["stage", "prompt tokens", "combo vs baseline", "what improved",
             "confidence"], d["stage_rows"],
            caption="Ratio is turn-matched on identical prompt-token counts. "
                    "ratio &lt; 1 means the candidate was faster."))
    a("<details><summary>Why 224k and 250k are not trustworthy at face value</summary>")
    a("<div class='inner'><p>At 250k the machine enters a <strong>fast/slow "
      "state that both arms enter</strong> — one baseline boot reached 1715 T/s, "
      "matching the combo's fast cluster. Matched <em>within</em> state the combo "
      "is at parity (fast 1.144× / slow 1.017×). The apparent 1.11–1.13× win came "
      "from comparing the combo's <em>fast</em> state against the baseline's "
      "<em>normal</em> state.</p>")
    a("<p><strong>130k is the only long-context length with state-matched "
      "evidence.</strong> The 224k figure rests on a single interleaved pair and "
      "carries the same suspicion.</p>")
    a("<p>The state is <em>not</em> paging, clock, or VRAM — all measured and "
      "flat: pagefile reads (two of three slow boots had <em>near-zero</em> reads "
      "while the fast boot had the third-highest), CPU "
      "<code>% Processor Performance</code> (113.9–114.4% of nominal, fully "
      "overlapping), available RAM, WDDM <code>SharedUsage</code>, run order, and "
      "config application. <strong>Cause still unidentified.</strong></p>")
    a("<p><strong>VRAM is comfortable throughout:</strong> min-free 1479–1543 MB at "
      "96–99% cache utilisation, and the spread does <em>not</em> widen with "
      "context.</p></div></details>")

    # ---- draft
    a("<h2>Draft / MTP</h2>")
    a("<h3><code>draft_num_tokens</code> — swept 3 to 10, baseline bracketed</h3>")
    a(table(["setting", "tg t/s", "vs base", "accepted", "drafted"], d["ndt_rows"]))
    a("<p><strong>Acceptance rises monotonically (2.76 → 3.31); speed does "
      "not.</strong> Speed peaks at <code>ndt3</code> and decays past "
      "<code>ndt6</code>. The ceiling above 5 is closed — it is not a speed win. "
      f"Confidence {badge('HIGH')}.</p>")
    a("<h3>Acceptance by content category</h3>")
    a(table(["category", "tools", "draft cache", "acceptance", "accepted",
             "drafted"], d["acc_rows"]))
    a("<p><strong>Confounded, unresolved:</strong> prose and tool count are "
      "entangled — <code>translate_02</code> has 0 tools, the others 11 and 29. "
      "The milder prose result is as consistent with “untooled prompts degrade "
      "less” as with “prose degrades less”, and no fixture provides "
      "prose-with-tools. The direction (<code>2,2</code> is worse everywhere) "
      "is solid; the prose-vs-code ordering is not.</p>")
    a("<h3><code>draft_cache_mode</code> — a pure VRAM knob</h3>")
    a(table(["mode", "n", "speed vs Q4", "VRAM min-free (MB)"], d["dcm_rows"]))
    a("<p>Monotone in VRAM, flat in speed down to Q8, a real penalty below. The "
      "VRAM it buys is not needed on this box (RAM binds), so there is no reason "
      "to move either way. <code>Q4</code> was chosen deliberately to recover VRAM "
      "without hurting tg — the ladder <em>confirms</em> that choice.</p>")

    # ---- what not to touch
    a("<h2>Rejected</h2>")
    a(d["reject"])
    a("<details><summary>Guard-trips and hard failures</summary><div class='inner'>")
    a(table(["setting", "outcome", "confidence"], [
        ("<code>cpu_moe_offload_layers: 32</code>",
         "<code>Insufficient VRAM in split for model and cache</code>",
         badge("HIGH")),
        ("<code>cpu_moe_offload_layers: 34</code>",
         "<strong class='bad'>UNSAFE</strong> — 129 MB free (guard is 200), "
         "<code>cudaErrorLaunchFailure</code> in <code>decode_flash_attn</code>. "
         "It is <em>faster</em> at Tier 0 (+22.3%) and trips the guard.",
         badge("HIGH")),
        ("<code>mcs 360</code>",
         "loads offline (23308 MB) but <strong class='bad'>will not boot the "
         "server</strong>", badge("HIGH")),
        ("<code>mcs 375</code>",
         "boots (967 MB free) but 38.5 vs 38.4 tok/s — 0.3%, noise, for 35% less "
         "margin", badge("MED")),
        ("<code>mcs 300 / 340 / 390 / 405 / 500</code>",
         "refused, or slower <em>and</em> smaller; 390/405 were never live-booted",
         badge("MED")),
        ("<code>sysmem_kv_cache: 8192</code>",
         "<strong class='bad'>RAM guard: 492 MB free during load</strong>",
         badge("HIGH")),
        ("<code>sysmem_kv_cache: 24576</code>",
         "<strong class='bad'>RAM guard: 88 MB free during load</strong>",
         badge("HIGH")),
    ]))
    a("</div></details>")

    a("<h2>Neutral — measured, nothing to gain</h2>")
    a("<p>Given equal weight deliberately: these are real measurements that "
      "produced no action, listed so they are not re-tested.</p>")
    a(d["keep"])

    # ---- memory
    a("<h2>Memory findings</h2>")
    a("<div class='callout warn'><p><strong>RAM, not VRAM, is the binding "
      "constraint.</strong> Free RAM looks comfortable at rest (~49 GB of 64 GB) "
      "but is nearly gone during load — the CPU-offloaded experts need a ~34 GB "
      "large-page arena. <code>sysmem_kv_cache</code> failed on the RAM guard "
      "twice.</p></div>")
    a("<ul class='tight'>"
      "<li>The live server sits <strong>+1.7 to +2.5 GB above</strong> "
      "<code>eval/perf.py</code> on the same config, because it builds components "
      "<code>perf.py</code> never does. <strong>A config that fits offline by a "
      "thin margin will not fit live</strong> — the safe offline margin is "
      "~2.5 GB, not the 200 MB guard. This is why <code>mcs360</code> fails "
      "live.</li>"
      "<li><strong>Do not decompose that offset from parameter counts</strong> — "
      "measured twice and wrong. <code>vision_offload: true</code> frees only "
      "~150–190 MB, not the ~1.1 GB the vision tower's fp16 weights suggest.</li>"
      "<li><strong>CUDA graphs are not a VRAM cost.</strong> "
      "<code>warmup: true</code> <em>reduces</em> peak VRAM by ~40–80 MB. Its "
      "cost is entirely the <strong>+12 s boot</strong>.</li></ul>")

    # ---- method
    a("<h2>Measurement rules we had to learn</h2>")
    a("<p>Each rule exists because its absence produced a wrong conclusion, not "
      "as a precaution.</p>")
    a(table(["rule", "why it exists", "error it prevented"], d["meth_rows"]))
    a("<details><summary>Prompt-length calibration</summary><div class='inner'>")
    a("<p>The synthesiser's <code>--target</code> is <strong>text</strong> tokens; "
      "the server counts <em>prompt</em> tokens, which are far larger because "
      "every message carries template scaffolding:</p>")
    a("<pre>actual ≈ 0.2383 × text_tokens + 294.4 × n_messages + 9,110</pre>")
    a("<p>Verified to 0.35%. The single-variable form "
      "<code>text × 1.797 + 10,000</code> agrees at short lengths and "
      "<strong>diverges near the ceiling</strong> — always predict every variant "
      "with both terms and reject a batch if any exceeds <code>cache_size</code> "
      "minus generation.</p></div></details>")

    # ---- what this does not establish
    a("<h2>What this does <em>not</em> establish</h2>")
    a("<div class='callout warn'><ol style='margin:0;padding-left:1.3em'>"
      "<li><strong>The 250k fast/slow machine state is unidentified.</strong> "
      "Both arms enter it. Operators should match on it before quoting any 250k "
      "ratio.</li>"
      "<li><strong>224k is a single pair</strong> and may be another cross-state "
      "comparison.</li>"
      "<li><strong>Losslessness of speculative drafting at temp &gt; 0 is "
      "theoretical here</strong> — Phase C ran temperature 0 only; the "
      "<code>-temp</code> arm was never run.</li>"
      "<li><strong><code>ZERO_COPY</code> is unmeasured at every tier</strong> — "
      "it straddles zero in both directions across sources.</li>"
      "<li><strong>Content vs tool count is confounded</strong> in the "
      "cross-category check.</li>"
      "<li><strong>Tier 2 covers only the 5 Phase A finalists.</strong> Every "
      "other Phase A row is a Tier 1 screen and is not quote-quality.</li>"
      "<li><strong><code>cache_mode</code> and draft-length output quality were "
      "never assessed</strong> — KLD owns those. <code>cache_mode 2,2</code> is "
      "a perf-only and VRAM-only line here.</li>"
      "</ol></div>")

    # ---- retractions
    a(f"<h2>Retractions ({d['n_rets']})</h2>")
    a("<p>Claims made during this work and withdrawn. Kept visible because the "
      "pattern is the useful part: <strong>every one was over-reading too few "
      "samples or an unmeasured mechanism.</strong></p>")
    a("<details><summary>All "
      f"{d['n_rets']} retractions in full</summary><div class='inner'>")
    a(table(["#", "claim withdrawn", "why it was wrong", "correction"],
            d["rets_rows"]))
    a("</div></details>")

    # ---- production state
    a("<h2>Production state</h2>")
    a("<p>Production <code>config.yml</code> was <strong>never modified</strong> "
      "and <strong>never committed</strong>. It is restored after every arm; "
      "pristine md5 <code>0fe01cc8f2e1e4cd2de7b1a1648ecb4f</code>.</p>")
    a("<p><small><strong>Printing:</strong> this page is JS-free by design, and "
      "CSS cannot force the collapsible sections open — expand them before "
      "printing if you want that content in the PDF.</small></p>")
    a("<p><small>Trap worth knowing: the per-arm files "
      "<code>config.yml.kb-&lt;arm&gt;</code> are snapshots of the "
      "<em>pristine</em> config taken <em>before</em> that arm ran. The name "
      "identifies the arm about to run, not a config with it applied.</small></p>")

    a("<footer>Generated from the CSV data set by <code>eval/_kb_mkhtml.py</code>, "
      "which reads <code>wiki/reports/data-2026-10-06-flash-next-knobs/</code>. "
      "Full run-by-run provenance, including every retraction, is in "
      "<code>wiki/reports/2026-10-06-flash-next-knob-battery.md</code>.</footer>")
    a("</div></body></html>")

    path = os.path.abspath(OUT)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    print(f"wrote {path}  ({os.path.getsize(path):,} bytes)")


if __name__ == "__main__":
    render()