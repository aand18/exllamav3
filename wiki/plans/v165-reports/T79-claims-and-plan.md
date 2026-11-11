# T79 — Kimi's claims, their status, and the plan that follows

Companion to `artifacts/reviews/T78-kimi-second-review-verbatim.md` (the review in full, so the
reasoning survives the summary) and `T78-kimi-second-pass.md` (what was already repaired).

## Part A — claims and status

| # | claim | status |
|---|---|---|
| **O1** | Response is substantive, but "all findings addressed" is not accurate | **accepted**; T78 opens by saying so |
| **O2** | No systematic misrepresentation of its review | noted, no action |
| **O3** | Adopted its vocabulary while retaining stronger conclusions | **accepted**; five instances corrected in T78 |
| **1a** | "Decode exhausted" is broader than the evidence | **fixed** (STATE + T77 §1) |
| **1b** | "Removing the fourth draft loses more than it saves" exceeds T70's evidence | **fixed** — withdrawn for material acceptance loss (−0.423), net throughput unresolved |
| **1c** | STATE still said "all zero / both closed / count lever closed" | **fixed** — "not detected at these doses … not an empirical closure" |
| **1d** | Decision closure is legitimate; empirical closure is not | **adopted as a wording rule** |
| **2a** | 3.70-of-4 and the 31 ms round are correct | **recorded** in STATE |
| **2b** | Provenance of 57 tok/s / 2.88 tok-per-round / 47.2 ms is unnamed and may mix drafter regimes | **OPEN** |
| **3a** | Nonlinear ladder recomputation is valuable | **recorded** |
| **3b** | 33.9 µs is a ratio of medians, not a causal marginal coefficient | **fixed** in STATE |
| **3c** | T69's failure to reproduce the ladder does not strengthen "count lever closed" | **fixed** — family "not detected" |
| **3d** | Payload does not "survive intervention"; no byte-width intervention has run | **fixed** — "untested, not validated" |
| **4a** | Static-placement constraint is properly represented | **recorded** |
| **4b** | The experiment recipe must repeat `SWAP=0` **and** `PROMOTE=0`, not just "static placement" | **OPEN** |
| **4c** | 26 % lower latency ≈ 35 % higher round rate | **recorded** |
| **4d** | The low-to-mid-70s projection must stay explicitly conditional | **fixed** in STATE |
| **5a** | Spread and primary used different estimators — my fix was incomplete | **fixed in T78** — one `primary_from_rows()` for effect and spread |
| **5b** | Variance combination was primary-only; category guards still used `max()` | **fixed** — categories now `sqrt(a²+t²)` |
| **5c** | Ratio scale asymmetric (`1 ± floor`) | **fixed** — `1/(1+floor)` on both sides |
| **5d** | T77's table shows `1+floor` while the prose discusses `1−floor` | **fixed** — docstring defines both |
| **5e** | Basis guard used `any()` while the estimator required `all()`; partial columns could mix bases | **fixed** — all-rows rule, per-row pairing, refusal tested |
| **5f** | No focused regression tests for the repairs | **fixed** — 5 new tests, 18 total |
| **5g** | Module docstring still described the superseded rules | **fixed** — current rules + superseded list |
| **6a** | The ten procedural rules are faithfully recorded | **recorded** |
| **6b** | Recorded but inconsistently applied (rule 1 breached; rule 10 would be breached by another ladder; rule 5 clashes with unnamed regimes) | **partly fixed** — rule 1 done; 5 and 10 **OPEN** |
| **7a** | Experiment and stopping rule are faithfully recorded, including the falsification logic | **recorded** |
| **7b** | "Quality-gated" needs a named numerical limit; "full mechanism success" must be tied to measured predictions | **OPEN** |
| **§3** | Most serious omission: unpriced `suh`/`svh` (~950 MiB); fixed-trace runner is the larger scientific one | **accepted; neither done** |
| **§4** | Gate priorities: finish estimator/basis → workload validity + fixed trace → per-category acceptance → more replication → noise-aware vetoes | **item 1 done; 2–5 OPEN** |
| **§5** | Authorise compile-only static falsification build, not a GPU A/B, until sizing is fixed; the required record sentence | **recorded** in T78 |

Totals: **20 addressed, 8 open, 1 recorded as a rule**.

## Part B — the plan

| # | action | reason | cost |
|---|---|---|---|
| **1** | Price `suh`/`svh` (and biases) in `bench/size_check.py`, so aux VRAM appears before a split is chosen | Kimi's ranked #1 omission; ~950 MiB unpriced has already caused two VRAM-floor breaches and is the stated blocker for any 2 bpw A/B. Rule 10 also forbids another ladder until it is done | ~1 h, no GPU |
| **2** | Put the required record sentence and a **named numerical quality limit** into the 2 bpw spec; tie "full mechanism success" to the pre-registered payload/`cmb`/round numbers | Without it a build can start and finish with its pass criterion chosen afterwards; Kimi 7b | ~30 min |
| **3** | Name the provenance of the projection inputs (arm, cache width, drafter) or re-derive them from one same-regime control | The 3.70-of-4 target and the 70s projection price the whole experiment; if they mix BF16 and exl3 regimes the target is wrong; Kimi 2b, and rule 5 | ~1 h, no GPU |
| **4** | Add a **per-category acceptance guard** to the gate | Aggregate acceptance hides an Agentic-curl regression — the category that moved most in T61 (−5.74 %) and T70 (−0.87 % with −0.35 acc) | ~1 h + tests |
| **5** | Add `EXL3_MOE_CPU_SWAP=0` / `EXL3_MOE_CPU_PROMOTE=0` assertions to the experiment recipe text | Trivial; prevents the build-time discovery Kimi flagged in 4b | minutes |
| **6** | Build the **fixed-trace systems runner** (replay fixed tokens/rows, gate on `ms/round`) | The largest scientific omission: without it a 2 bpw A/B cannot separate payload from routing, and every systems arm keeps riding 3–7 % trajectory noise | days — the one real project left |
| **7** | Then the compile-only 2 bpw build, gated on 1–3 | Its authorisation is explicitly conditional on sizing, a quality limit and provenance | after 1–3 |

**Order: 5 → 2 → 1 → 3 → 4, then choose between 6 and 7.**

Rationale for the ordering: item 5 is minutes and removes a documented trap. Item 2 defines what a
success even means. Items 1 and 3 make the experiment safe and correctly priced. Item 4 closes the
last mechanism-blind spot in the gate. Items 6 and 7 are the two-week decisions — 6 changes how every
future systems experiment is judged, 7 is the only remaining lever that could move tok/s by more than
noise.

## Part C — what this plan does not do

- It does not reopen the MoE handoff, placement, split or pinning families: those are "not detected at
  these doses", and re-running them at larger doses is a poor-ROI decision recorded in STATE.md.
- It does not promise the 2 bpw arm reaches 93 tok/s. The honest conditional projection is
  low-to-mid 70s, and the falsification logic is explicit: if payload falls ~37 % and `cmb` barely
  moves, the bytes law is falsified — that is new information, not exhaustion.
- It does not treat the methodology as settled. The largest external weakness remains workload
  validity: live acceptance 1.58 of 4 against a corpus 2.07, which no gate repairs.
