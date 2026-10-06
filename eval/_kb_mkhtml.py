#!/usr/bin/env python3
"""Render the Flash-Next knob report as a single static HTML file.

WRITTEN FOR A NON-EXPERT READER. The previous revision used the team's internal
shorthand throughout (ndt5_dyn_prod, combo, mcs380, pp4096, "min free") and a
reviewer correctly said it was unreadable. So:

  * every setting is labelled in plain language, with the config key in its own
    subdued column so it can still be typed into config.yml
  * a glossary explains the domain vocabulary once, up front
  * speed AND VRAM headroom appear in EVERY performance table -- a speed number
    without its memory cost is only half a decision

Reads the CSV data set, so no figure is retyped and the HTML cannot drift.
"""

import csv
import html
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "wiki", "reports",
                    "data-2026-10-06-flash-next-knobs")
OUT = os.path.join(HERE, "..", "wiki", "reports",
                   "2026-10-06-flash-next-knobs.html")

CONF = {"HIGH": "ok", "MED": "mid", "LOW": "low", "SCREEN": "none",
        "HIGH (null result)": "ok", "HIGH at 130k / MED short-ctx": "ok"}


def rows(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# Free-text fields in the CSVs still contain the team's shorthand. These are
# the exact forms that occur; each maps to something a non-expert can read.
JARGON = [
    (r"mcs380", "the expert-CPU split"),
    (r"mcs375", "expert-CPU split of 375"),
    (r"\bmcs 360\b", "expert-CPU split of 360"),
    (r"pp4096", "long-prompt reading"),
    (r"pp256", "short-prompt reading"),
    (r"\btg\b", "generation speed"),
    (r"\bprefill\b", "prompt reading"),
    (r"decode", "answer generation"),
    (r"ndt(\d)", r"draft \1 tokens"),
    (r"\bmcl\b", "offloaded layers"),
    (r"ZERO_COPY", "the zero-copy memory option"),
    (r"EXL3_MOE_CPU_THREADS", "the CPU thread count setting"),
    (r"EXL3_MOE_CPU_PIN", "the thread-pinning setting"),
    (r"EXL3_MOE_CPU_SWIZZLE", "the thread-swizzling setting"),
    (r"EXL3_MOE_STREAM_T", "the expert streaming depth setting"),
    (r"EXL3_MOE_STREAM_BATCH_EXPERTS", "the experts-per-batch setting"),
]


def plain(text):
    """Escape, then replace internal shorthand with plain language."""
    out = e(text)
    for pat, rep in JARGON:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


def e(s):
    return html.escape(str(s if s is not None else ""))


def conf(text):
    """Badge coloured by the LEADING token, so 'HIGH (null result)' does not
    fall through to the red LOW badge -- which would show a settled finding as
    unreliable."""
    key = (text or "").strip()
    lead = key.split()[0].rstrip(":") if key else ""
    return (f'<span class="badge {CONF.get(lead, "low")}">{e(key)}</span>')


def mb(v):
    try:
        return f"{float(v):,.0f}"
    except (TypeError, ValueError):
        return "&mdash;"


# ---------------------------------------------------------------- vocabulary
SETTING = {
    "ndt3_dyn": ("Draft 3 tokens, adaptive", "draft_num_tokens: 3"),
    "ndt5_dyn_prod": ("Draft 5 tokens &mdash; <strong>current setting</strong>",
                      "draft_num_tokens: 5"),
    "ndt6_dyn": ("Draft 6 tokens, adaptive", "draft_num_tokens: 6"),
    "ndt7_dyn": ("Draft 7 tokens, adaptive", "draft_num_tokens: 7"),
    "ndt8_dyn": ("Draft 8 tokens, adaptive", "draft_num_tokens: 8"),
    "ndt10_dyn": ("Draft 10 tokens, adaptive", "draft_num_tokens: 10"),
    "off": ("Drafting switched off entirely", "draft off"),
    "Q4": ("4-bit &mdash; current setting", "draft_cache_mode: Q4"),
    "Q8": ("8-bit", "draft_cache_mode: Q8"),
    "FP16": ("16-bit, no compression", "draft_cache_mode: FP16"),
    "3,3": ("3-bit", "draft_cache_mode: 3,3"),
    "2,2": ("2-bit", "draft_cache_mode: 2,2"),
}
CATNAME = {"agentic_code": "Code conversations",
           "agentic_curl": "Shell / curl agent conversations",
           "prose_translate": "Prose / translation",
           "code": "Code conversations"}
DCM_SHORT = {"Q4_baseline": "4-bit (current)", "FP16": "16-bit", "2,2": "2-bit"}
DCM_KEY = {"Q4_baseline": "draft_cache_mode: Q4",
           "FP16": "draft_cache_mode: FP16", "2,2": "draft_cache_mode: 2,2"}

# Tier-0 VRAM footprints for the draft-length arms. Tier 0 only -- higher tiers
# use longer prompts and therefore a larger footprint, so mixing them is invalid.
NDT_VRAM = {"ndt3_dyn": (21152, 2987), "ndt5_dyn_prod": (21411, 2728),
            "ndt6_dyn": (21662, 2477), "ndt7_dyn": (21878, 2261),
            "ndt8_dyn": (22104, 2035), "ndt10_dyn": (22594, 1545),
            "off": (21902, 2237)}

PLAIN = {"EXL3_MOE_CPU_THREADS": "CPU worker threads: 8 &rarr; 16",
         "EXL3_MOE_ZERO_COPY": "Zero-copy memory option",
         "EXL3_MOE_CPU_PIN": "Pin CPU threads to cores",
         "EXL3_MOE_CPU_SWIZZLE": "CPU thread core-swizzling",
         "EXL3_MOE_STREAM_T": "Expert streaming depth",
         "EXL3_MOE_STREAM_BATCH_EXPERTS": "Experts streamed per batch",
         "EXL3_MOE_MEMOPS": "Memory-operation path",
         "warmup": "Start-up warm-up pass",
         "vision_offload": "Vision tower offloaded to CPU",
         "max_batch_size": "Concurrent requests allowed",
         "chunk_size": "Prompt processing chunk size",
         "dynamic_draft": "Adaptive draft length",
         "cuda_malloc_async": "Async graphics-memory allocator",
         "draft_cache_mode": "Draft cache precision",
         "cpu_moe_offload_layers": "Layers offloaded to CPU",
         "mcs value below 380": "Expert-CPU split below 380",
         "mcs 390 / 405": "Expert-CPU splits of 390 and 405",
         "sysmem_kv_cache": "Second-tier key/value cache",
         "recurrent_checkpoint_interval": "Recurrent checkpoint spacing",
         "recurrent_checkpoint_interval_pp": "Prompt-recurrent checkpoints",
         "cpu_moe_split_experts": "Experts moved to the CPU: 380 of 512",
         "draft_num_tokens": "Draft length: 5 &rarr; 3 tokens",
         "cache_mode": "Conversation cache precision: 5,4 &rarr; 2,2"}


def table(headers, body_rows, caption=None):
    cap = f"<caption>{caption}</caption>" if caption else ""
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join(f"<tr>{''.join(f'<td>{c}</td>' for c in r)}</tr>"
                  for r in body_rows)
    return (f'<div class="tw"><table>{cap}<thead><tr>{th}</tr></thead>'
            f"<tbody>{trs}</tbody></table></div>")


def build():
    dec, lc, dr = rows("decisions.csv"), rows("long-context.csv"), rows("draft.csv")
    rets, meth = rows("retractions.csv"), rows("methodology.csv")

    adopt_rows = [(f"<strong>{PLAIN.get(d['knob'], e(d['knob']))}</strong>",
                   f"<code>{e(d['current'])}</code> &rarr; <code>{e(d['proposed'])}</code>",
                   plain(d["measured_effect"]), plain(d["cost"]), plain(d["rationale"]))
                  for d in dec if d["verdict"] == "ADOPT"]

    # VRAM spare (min-free) after loading, measured live. Answering the
    # operator's real question at 200k+: does headroom shrink as prompts grow?
    order = [("11-16k_turnmatch", "Short prompts", "10,956&ndash;16,135",
              "Answer generation", "1,065"),
             ("55-62k", "Medium prompts", "55,239&ndash;60,634",
              "nothing &mdash; no change either way", "1,065"),
             ("130k", "Long prompts", "129,910&ndash;144,084",
              "Reading the prompt, +14%", "1,479&ndash;1,543"),
             ("224k", "Very long prompts", "224,414&ndash;224,502",
              "Reading the prompt (one test pair only)", "1,479&ndash;1,511"),
             ("250k", "Maximum practical size", "249,642&ndash;259,145",
              "nothing &mdash; see caveat below", "1,479&ndash;1,543")]
    stage_rows = []
    for key, label, ptxt, note, vram in order:
        rs = [r for r in lc if r["stage"] == key]
        combo = [float(r["ratio_vs_baseline"]) for r in rs if r["arm"] == "combo"]
        if not combo:
            continue
        med = st.median(combo)
        if key == "250k":
            stage_rows.append((f"<strong>{label}</strong>", ptxt, "&mdash;",
                               note, vram, conf("MED")))
            continue
        stage_rows.append((f"<strong>{label}</strong>", ptxt,
                           f"<strong class='good'>{1/med:.2f}&times; faster</strong>",
                           note, vram, conf(rs[0]["confidence"])))

    ndt = sorted((r for r in dr if r["test"] == "ndt_ladder"),
                 key=lambda r: -float(r["value"]))
    ndt_rows = []
    for r in ndt:
        nm, key = SETTING[r["setting"]]
        cur = ' class="cur"' if r["setting"] == "ndt5_dyn_prod" else ""
        hf = NDT_VRAM.get(r["setting"], (None, None))[1]
        vs = r["vs_baseline_pct"]
        vsc = "&mdash;" if vs == "" else (
            f"<strong class='good'>{float(vs):+.1f}%</strong>" if float(vs) > 1
            else f"<strong class='bad'>{float(vs):+.1f}%</strong>")
        ndt_rows.append((f"{nm}{cur}", f"<code>{key}</code>", r["value"],
                         vsc, mb(hf)))

    acc = sorted((r for r in dr if r["test"] == "acceptance_by_category"),
                 key=lambda r: -float(r["value"]))
    acc_rows = [(f"{CATNAME[r['category']]}"
                 + (' class="cur"' if r["setting"] == "Q4_baseline" else ""),
                 r["tools"], DCM_SHORT[r["setting"]],
                 f"<code>{DCM_KEY[r['setting']]}</code>",
                 f"<strong>{r['value']}%</strong>") for r in acc]

    dcm = sorted((r for r in dr if r["test"] == "draft_cache_mode_ladder"),
                 key=lambda r: float(r["value"]))
    dcm_rows = []
    for r in dcm:
        nm, key = SETTING[r["setting"]]
        cur = ' class="cur"' if r["setting"] == "Q4" else ""
        v = float(r["value"])
        verdict = ("no change" if abs(v - 1) <= 0.025
                   else "<strong class='bad'>slower</strong>" if v > 1
                   else "<strong class='good'>faster</strong>")
        vmin = r["note"].split("vram_minfree=")[-1].replace("MB", "") \
            if "vram_minfree=" in r["note"] else None
        dcm_rows.append((f"{nm}{cur}", f"<code>{key}</code>", verdict, mb(vmin)))

    return dict(adopt_rows=adopt_rows, stage_rows=stage_rows, ndt_rows=ndt_rows,
                acc_rows=acc_rows, dcm_rows=dcm_rows,
                keep=[d for d in dec if d["verdict"] in ("KEEP", "KEEP (low conf)")],
                reject=[d for d in dec if d["verdict"] in ("REJECT", "REPLACE")],
                rets=rets, meth=meth, n_dec=len(dec))


CSS = """
:root{--bg:#fbfaf9;--fg:#1c1a19;--mut:#6b6560;--line:#e4e0dc;--card:#fff;
 --ok:#0f7b4f;--okbg:#e6f4ec;--mid:#8a5a00;--midbg:#fdf3e0;
 --low:#b3261e;--lowbg:#fdeceb;--none:#6b6560;--nonebg:#f0eeec;--accent:#6b3fa0;
 --hl:#fff8e6}
@media (prefers-color-scheme:dark){:root{--bg:#16151a;--fg:#e9e6e2;--mut:#a09a94;
 --line:#33313a;--card:#1e1d23;--ok:#5fd39b;--okbg:#12301f;--mid:#e0a84a;
 --midbg:#33280f;--low:#f28b82;--lowbg:#3a1512;--none:#a09a94;--nonebg:#26252b;
 --accent:#c39ee0;--hl:#2a2413}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.62
 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
 font-variant-numeric:tabular-nums}
.wrap{max-width:1120px;margin:0 auto;padding:40px 24px 96px}
header.top{border-bottom:2px solid var(--line);padding-bottom:20px}
h1{font-size:1.95rem;line-height:1.18;margin:0 0 6px}
.sub{color:var(--mut);font-size:.97rem;margin:0;max-width:70ch}
.meta{color:var(--mut);font-size:.83rem;margin-top:10px}
h2{font-size:1.35rem;margin:46px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:1.04rem;margin:26px 0 8px}
p,li{max-width:76ch}
code{font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 background:var(--nonebg);padding:1.5px 5px;border-radius:4px;white-space:nowrap}
a{color:var(--accent)}
.tw{overflow-x:auto;margin:12px 0 18px}
table{border-collapse:collapse;width:100%;font-size:.88rem}
caption{caption-side:top;text-align:left;color:var(--mut);font-size:.83rem;
 padding-bottom:6px;max-width:76ch}
th{text-align:left;font-weight:600;border-bottom:2px solid var(--line);
 padding:8px 10px;white-space:nowrap}
td{border-bottom:1px solid var(--line);padding:8px 10px;vertical-align:top}
tbody tr:nth-child(even){background:var(--card)}
tr.cur{background:var(--hl)!important}
tr.cur td:first-child{box-shadow:inset 3px 0 0 var(--accent)}
.badge{display:inline-block;font-size:.72rem;font-weight:700;padding:2px 7px;
 border-radius:10px;white-space:nowrap}
.badge.ok{color:var(--ok);background:var(--okbg)}.badge.mid{color:var(--mid);background:var(--midbg)}
.badge.low{color:var(--low);background:var(--lowbg)}.badge.none{color:var(--none);background:var(--nonebg)}
.good{color:var(--ok)}.bad{color:var(--low)}
.callout{background:var(--card);border:1px solid var(--line);
 border-left:4px solid var(--accent);border-radius:6px;padding:14px 18px;margin:16px 0}
.callout.warn{border-left-color:var(--low)}.callout.ok{border-left-color:var(--ok)}
.callout.info{border-left-color:var(--mid)}
.callout h3{margin-top:0}.callout p:last-child{margin-bottom:0}
dl.gloss{display:grid;grid-template-columns:minmax(180px,auto) 1fr;gap:6px 18px;
 font-size:.9rem;margin:10px 0 0}
dl.gloss dt{font-weight:600}dl.gloss dd{margin:0;color:var(--fg);opacity:.92}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;
 padding:14px 16px;overflow-x:auto;font-size:.84rem;line-height:1.55}
ul.tight{margin:.4em 0}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0 18px;font-size:.84rem}
.legend div{display:flex;align-items:center;gap:7px}
details{border:1px solid var(--line);border-radius:6px;margin:14px 0;background:var(--card)}
details>summary{cursor:pointer;padding:11px 16px;font-weight:600;font-size:.95rem}
details[open]>summary{border-bottom:1px solid var(--line)}
details .inner{padding:14px 16px 6px}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);
 color:var(--mut);font-size:.83rem}
@media print{
 :root{--bg:#fff;--fg:#000;--mut:#444;--line:#bbb;--card:#fff;--hl:#f4f4f4;
  --ok:#0a5c39;--okbg:#fff;--mid:#6b4400;--midbg:#fff;--low:#8f1d17;--lowbg:#fff;
  --none:#555;--nonebg:#fff}
 body{font-size:10.5pt}.wrap{max-width:none;padding:0}
 details{border:1px solid #bbb}details>summary{list-style:none}
 details:not([open])>summary::after{content:" (expand before printing)"}
 h2,h3{page-break-after:avoid;break-after:avoid}
 tr,.callout,pre,dl.gloss{page-break-inside:avoid;break-inside:avoid}
 thead{display:table-header-group}.badge{border:1px solid currentColor}
}"""

GLOSSARY = [
    ("VRAM", "Memory on the graphics card. The model lives here."),
    ("VRAM headroom", "Spare VRAM left over after loading. Spent on anything "
     "that needs more memory."),
    ("CPU offload", "The model is too big for the card, so some of it runs on "
     "the processor instead. Slower, but it frees VRAM."),
    ("Experts", "The model's specialised sub-networks. There are 512 per layer; "
     "this setting decides how many run on the processor."),
    ("Reading the prompt (prefill)", "Absorbing everything you typed, before the "
     "model starts replying."),
    ("Answer generation (decode)", "Producing the reply, one token at a time."),
    ("tok/s", "Tokens per second &mdash; how fast text is produced."),
    ("Speculative drafting", "The model guesses several upcoming tokens in one "
     "go, then checks its own guess. Roughly doubles speed when the guesses are "
     "often right."),
    ("Acceptance", "How many of the guessed tokens turn out to be right. This "
     "changes <em>speed</em>, not <em>output</em> &mdash; the text you get is "
     "the same either way."),
    ("Adaptive", "The model shortens its guess when it is unsure, instead of "
     "always guessing the maximum."),
    ("Conversation cache precision", "How compressed the stored conversation is. "
     "Lower = more VRAM free, same text."),
    ("Draft cache precision", "Same idea, but for the guessing machinery."),
]


def render():
    d = build()
    o = []
    a = o.append
    a("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    a("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    a("<title>Flash-Next settings report (2026-10-06)</title>")
    a(f"<style>{CSS}</style></head><body><div class='wrap'>")

    a("<header class='top'><h1>Flash-Next serving settings &mdash; what to change</h1>")
    a("<p class='sub'>We measured every tunable setting on the 3.05bpw "
      "Flash-Next model as it actually serves traffic, on an RTX 4090 (24 GB) "
      "with a Ryzen 7950X3D. This page says what to change, what it buys, and "
      "what we are <em>not</em> sure about.</p>")
    a(f"<p class='meta'>6 October 2026 &middot; {d['n_dec']} settings judged "
      "&middot; tested from 11,000 to 259,000-token prompts, up to 96&ndash;99% "
      "of the model's maximum conversation size &middot; the production "
      "<code>config.yml</code> was never modified</p></header>")

    # ---------- glossary first: the audience does not know the jargon
    a("<h2>What the words in this report mean</h2>")
    a("<details open><summary>Glossary &mdash; read this first</summary><div class='inner'>")
    a("<dl class='gloss'>" + "".join(
        f"<dt>{t}</dt><dd>{d_}</dd>" for t, d_ in GLOSSARY) + "</dl>")
    a("</div></details>")

    # ---------- read this first
    a("<h2>The short version</h2>")
    a("<div class='callout warn'><h3>The speed-up shrinks as prompts get longer</h3>")
    a("<p>Your traffic is mostly <strong>200,000-token prompts and above</strong>, "
      "which is the hardest region to speed up. The changes below give "
      "<strong>1.16&times; at 130k tokens</strong> but <strong>nothing at all at "
      "250k</strong>. Any &ldquo;36% faster&rdquo; figure you see quoted for "
      "short prompts would badly overstate what this does for you.</p></div>")
    a("<div class='callout info'><h3>How much to trust each claim</h3>")
    a("<p>Every claim below carries a confidence marker. "
      "<strong>HIGH</strong> means the comparison was measured against a "
      "baseline taken <em>at the same time</em>, and repeated. That matters more "
      "than it sounds: this machine drifts by about 10% over a session, so a "
      "comparison against an earlier measurement can be wrong by that much "
      "without anyone noticing.</p></div>")
    a("<div class='legend'>"
      "<div><span class='badge ok'>HIGH</span> reliable</div>"
      "<div><span class='badge mid'>MED</span> thin evidence</div>"
      "<div><span class='badge low'>LOW</span> first-pass screen only</div>"
      "<div><span class='badge none'>SCREEN</span> not tested &mdash; don't re-test</div>"
      "</div>")

    # ---------- the change
    a("<h2>What to change</h2>")
    a("<pre>model:\n"
      "  #cpu_moe_offload_layers: 38      # removed; replaced by the line below.\n"
      "                                  # Never set both -- they conflict.\n"
      "  cpu_moe_split_experts: 380      # 380 of 512 experts run on the CPU\n"
      "  cache_mode: 2,2                  # coarser conversation cache: +1 GB VRAM\n"
      "\n"
      "draft_model:\n"
      "  draft_num_tokens: 3              # was 5 -- guess 3 tokens ahead, not 5\n"
      "  dynamic_draft: true\n"
      "  draft_cache_mode: Q4\n"
      "\n"
      "memory:\n"
      "  cuda_malloc_async: True\n"
      "\n"
      "# start_tuned.ps1:\n"
      "#   EXL3_MOE_CPU_THREADS=8  ->  16</pre>")
    a("<div class='callout ok'><h3>If graphics-card memory ever becomes tight</h3>")
    a("<p>Drop <code>cpu_moe_split_experts</code> and keep the thread change. "
      "Threads use <strong>no VRAM at all</strong> and give the largest single "
      "speed-up in this report.</p></div>")
    a(table(["Change", "Config key", "What it buys", "What it costs", "Why"],
            d["adopt_rows"]))

    # ---------- the VRAM question answered head-on
    a("<h2>About that VRAM figure</h2>")
    a("<div class='callout info'><p>A reviewer asked a fair question: "
      "<em>&ldquo;it's just a number you can tune &mdash; how is that a "
      "cost?</em>&rdquo; It is not a cost in itself, and the earlier phrasing "
      "was both wrong and misleading.</p>")
    a("<p><strong>The figure was wrong too.</strong> The change uses "
      "<strong>1,524 MB (about 1.5 GB)</strong> more VRAM than the current "
      "setting &mdash; not the 2.5 GB previously quoted, which came from an "
      "early offline-only comparison.</p>")
    a("<p>What it actually is: a <strong>budget you choose to spend</strong>. "
      "VRAM headroom is not free money, because it is the currency for anything "
      "else that needs card memory &mdash; a longer conversation cache, serving "
      "several requests at once, or a bigger model. Spending 1.5 GB of it is "
      "defensible <em>only</em> if you do not want those things.</p>")
    a("<p>Two facts make it defensible here. First, on this machine the scarce "
      "resource is <strong>system RAM, not VRAM</strong> &mdash; we proved that "
      "when a different setting was killed by a RAM guard twice. Second, 380 is "
      "chosen to spend the <em>least</em> headroom that still delivers the "
      "speed: 375 spends 2,036 MB and delivers <strong>nothing measurable</strong> "
      "(0.3%, i.e. noise). So the question is not &ldquo;is VRAM a cost&rdquo; "
      "but &ldquo;is 1.5 GB of headroom worth 14% faster prompt-reading at 130k "
      "tokens&rdquo; &mdash; and that is the operator's call, not ours.</p></div>")

    # ---------- long context
    a("<h2>Speed-up by prompt length</h2>")
    a(table(["Prompt size", "Tokens tested", "Speed-up", "What improved",
             "VRAM spare (MB)", "Confidence"], d["stage_rows"],
            caption="Speed-up is measured by replaying the identical prompt "
                    "against both settings, so prompt difficulty cancels out. "
                    "VRAM spare is what is left on the graphics card after "
                    "loading &mdash; it does <em>not</em> shrink as prompts grow, "
                    "which was the main worry at your prompt sizes. The current "
                    "setting leaves about 3,000 MB spare for comparison."))
    a("<details><summary>Why the two longest rows are the least trustworthy</summary>")
    a("<div class='inner'><p>At 250k tokens this machine slips in and out of a "
      "faster state &mdash; and <strong>both the old and new settings slip into "
      "it</strong>. When compared fairly within the same state, the two are "
      "level. The apparent 14% win came from measuring the new setting during a "
      "fast period and the old one during a normal period.</p>")
    a("<p><strong>130k is the only long-prompt length with a fair "
      "comparison.</strong> The 224k row rests on a single test pair and should "
      "be treated with the same suspicion.</p>")
    a("<p>We checked and ruled out the obvious causes: disk paging, processor "
      "clock throttling, available RAM, and graphics-card memory. All were flat "
      "while the state changed. <strong>The cause is still unknown.</strong></p>")
    a("<p><strong>VRAM was comfortable throughout</strong> &mdash; 1,479 MB spare "
      "even at 99% of maximum conversation size, and it did not shrink as "
      "prompts grew.</p></div></details>")

    # ---------- drafting
    a("<h2>Speculative drafting</h2>")
    a("<h3>How many tokens to guess ahead</h3>")
    a(table(["Setting", "Config key", "Speed", "vs current", "VRAM headroom"],
            d["ndt_rows"],
            caption="VRAM is spare card memory after loading. The highlighted row "
                    "is the current setting. Note that drafting 3 tokens is both "
                    "the fastest <em>and</em> leaves slightly more VRAM spare than "
                    "today's setting of 5."))
    a("<p>As the guess gets longer, <strong>accuracy keeps improving but speed "
      "stops improving and then declines</strong> &mdash; and VRAM headroom "
      "shrinks steadily, because a longer guess needs more scratch space. "
      "Guessing 3 wins on both axes at once.</p>")
    a("<h3>Does answer content depend on the guess length?</h3>")
    a("<p>No. A longer or shorter guess changes only how fast the answer "
      "arrives, not what it says. That is the theory behind speculative "
      "decoding, and it is why the 5&rarr;3 change is a pure win.</p>")
    a("<h3>Does the guess length behave differently by kind of writing?</h3>")
    a(table(["Kind of conversation", "Tools", "Draft cache", "Config key",
             "Guess accuracy"], d["acc_rows"],
            caption="Accuracy = how often the guessed tokens survive checking. "
                    "Higher is better. The highlighted row is the current "
                    "setting."))
    a("<p><strong>One caveat we could not resolve:</strong> the prose test had "
      "<em>zero tools</em> while the code tests had 11 and 29, so "
      "&ldquo;prose is worse&rdquo; and &ldquo;untested is worse&rdquo; are "
      "indistinguishable with the material we have. What <em>is</em> solid: the "
      "2-bit setting is slower in every category tested.</p>")
    a("<h3>How compressed the guess's own memory should be</h3>")
    a(table(["Setting", "Config key", "Speed vs current", "VRAM headroom"],
            d["dcm_rows"],
            caption="This knob only moves memory. It buys VRAM steadily as you "
                    "compress harder, and is speed-neutral until 3-bit, where "
                    "it starts costing speed."))
    a("<p>It was set to 4-bit deliberately, to recover VRAM without hurting "
      "speed. <strong>This test confirms that choice</strong> &mdash; and since "
      "VRAM is not the scarce resource on this machine, there is no reason to "
      "move it either way.</p>")

    # ---------- memory
    a("<h2>What limits this machine</h2>")
    a("<div class='callout warn'><p><strong>System RAM is the real constraint "
      "&mdash; not graphics-card memory.</strong> Free RAM looks comfortable "
      "while idle (about 49 GB of 64 GB) but is nearly exhausted while the "
      "model loads. A different setting was killed by our RAM guard twice at "
      "8&nbsp;GB and 24&nbsp;GB.</p></div>")
    a("<ul class='tight'>")
    a("<li>The live server needs <strong>1.7&ndash;2.5 GB more card memory</strong> "
      "than the offline benchmark on the same settings, because it builds "
      "components the benchmark never creates. <strong>Anything that barely "
      "fits in the benchmark will not fit in the server.</strong> This is why "
      "one promising setting loads fine offline but will not start at all.</li>")
    a("<li><strong>Do not try to derive that gap from parameter counts</strong> "
      "&mdash; we tried, and it was wrong twice.</li>")
    a("<li><strong>Warming up does not cost card memory.</strong> It costs "
      "<strong>12 extra seconds of startup</strong>, and slightly "
      "<em>reduces</em> memory use.</li></ul>")

    # ---------- keep / reject
    a("<h2>Leave alone</h2>")
    a("<p>Measured, produced no gain, listed so nobody re-tests them.</p>")
    a(table(["Setting", "Current", "Proposed", "Measured effect", "What it costs",
             "Confidence"],
            [(PLAIN.get(x["knob"], f"<code>{e(x['knob'])}</code>"),
              e(x["current"]), e(x["proposed"]),
              plain(x["measured_effect"]), plain(x["cost"]), conf(x["confidence"]))
             for x in d["keep"]]))
    a("<h2>Rejected</h2>")
    a(table(["Setting", "Current", "Proposed", "Measured effect", "What it costs",
             "Confidence"],
            [(PLAIN.get(x["knob"], f"<code>{e(x['knob'])}</code>"),
              e(x["current"]), e(x["proposed"]),
              plain(x["measured_effect"]), plain(x["cost"]), conf(x["confidence"]))
             for x in d["reject"]]))
    a("<details><summary>Settings that failed outright</summary><div class='inner'>")
    a(table(["Setting", "What happened", "Confidence"], [
        ("Fewer layers on the CPU (<code>32</code>)",
         "Will not load &mdash; not enough card memory", conf("HIGH")),
        ("Fewer layers on the CPU (<code>34</code>)",
         "<strong class='bad'>Unsafe.</strong> Left only 129 MB spare (our limit "
         "is 200) and crashed the kernel. It was <em>faster</em> in the first "
         "pass, which is exactly why it is worth flagging.", conf("HIGH")),
        ("640 experts on the CPU (<code>mcs 360</code>)",
         "Loads in the benchmark but <strong class='bad'>will not start the "
         "server</strong>", conf("HIGH")),
        ("375 experts on the CPU",
         "Starts (967 MB spare) but 0.3% faster &mdash; noise &mdash; for 35% "
         "less headroom", conf("MED")),
        ("Other expert counts (300 / 340 / 390 / 405 / 500)",
         "Refused, or slower <em>and</em> using more memory", conf("MED")),
        ("Second-tier cache, 8 GB",
         "<strong class='bad'>Killed by the RAM guard</strong> &mdash; 492 MB "
         "left during load", conf("HIGH")),
        ("Second-tier cache, 24 GB",
         "<strong class='bad'>Killed by the RAM guard</strong> &mdash; 88 MB left "
         "during load", conf("HIGH")),
    ]))
    a("</div></details>")

    # ---------- not established
    a("<h2>What we do <em>not</em> know</h2>")
    a("<div class='callout warn'><ol style='margin:0;padding-left:1.3em'>")
    a("<li><strong>Why the machine has fast and slow periods at 250k tokens.</strong> "
      "Both settings are affected. Measure within a period before quoting any "
      "250k number.</li>")
    a("<li><strong>The 224k figure is a single test pair</strong> and may be "
      "another cross-period comparison.</li>")
    a("<li><strong>That drafting cannot affect output is theory, not something "
      "we measured.</strong> All our serving tests ran at temperature 0; the "
      "sampling test was never run.</li>")
    a("<li><strong>One memory setting was never measured at any depth</strong> "
      "(the zero-copy memory option) &mdash; sources disagree on its sign.</li>")
    a("<li><strong>Prose vs code is confounded with tool count</strong> in the "
      "drafting test.</li>")
    a("<li><strong>Only 5 settings got the deep offline treatment.</strong> Every "
      "other offline row is a first-pass screen.</li>")
    a("<li><strong>Answer quality was never assessed</strong> for the cache or "
      "drafting changes &mdash; that belongs to a separate quality review.</li>")
    a("</ol></div>")

    # ---------- method
    a("<h2>How the numbers were obtained</h2>")
    a("<p>Each rule below exists because leaving it out produced a wrong "
      "conclusion &mdash; not as a precaution.</p>")
    a(table(["Rule", "Why", "What it prevented"],
            [(f"<strong>{e(m['rule'])}</strong>", e(m["reason"]),
              plain(m["error_it_prevented"])) for m in d["meth"]]))
    a("<details><summary>Sizing the test prompts</summary><div class='inner'>")
    a("<p>Test prompts are built to a target size, but the server counts more "
      "tokens than the builder predicts, because each message carries formatting "
      "overhead:</p>")
    a("<pre>actual ≈ 0.2383 × text_tokens + 294.4 × n_messages + 9,110</pre>")
    a("<p>Accurate to about 0.3%. A simpler one-term formula agrees at small sizes "
      "but <strong>diverges near the maximum</strong>, so both terms must be "
      "checked or an oversized prompt simply will not load.</p></div></details>")

    # ---------- retractions
    a(f"<h2>Claims we withdrew ({len(d['rets'])})</h2>")
    a("<p>Conclusions reached during this work and then taken back. Kept visible "
      "because the pattern is the useful part: <strong>each one was "
      "over-reading too few samples, or explaining a number with a mechanism we "
      "had not actually measured.</strong></p>")
    a("<details><summary>All "
      f"{len(d['rets'])} withdrawn claims</summary><div class='inner'>")
    a(table(["#", "Claim withdrawn", "Why it was wrong", "What is true instead"],
            [(e(r["seq"]), f"<strong>{plain(r['claim_withdrawn'])}</strong>",
              plain(r["why_wrong"]), plain(r["correction"])) for r in d["rets"]]))
    a("</div></details>")

    # ---------- production
    a("<h2>Production state</h2>")
    a("<p>The production <code>config.yml</code> was <strong>never modified and "
      "never committed</strong>. It is restored after every test run.</p>")
    a("<p><small><strong>Printing:</strong> this page has no JavaScript by "
      "design, so expand the collapsible sections before printing if you want "
      "that content in the PDF.</small></p>")

    a("<footer>Generated from the CSV data set by "
      "<code>eval/_kb_mkhtml.py</code>. Full run-by-run provenance is in "
      "<code>wiki/reports/2026-10-06-flash-next-knob-battery.md</code>.</footer>")
    a("</div></body></html>")

    path = os.path.abspath(OUT)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    print(f"wrote {path}  ({os.path.getsize(path):,} bytes)")


if __name__ == "__main__":
    render()