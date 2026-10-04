# Subagent report: Self-improvement papers hunt (2026-10-01)

## Provenance
RECOVERED from the opencode session store (`ses_f0b3264f6ffep8oam9zOVAMrN5`, dispatched
2026-10-01 00:11 UTC): verbatim final-report text below (the subagent's prompts
capped these at ~400 words / 450 words, so this IS the complete
deliverable, not a compression). Pre-convention dispatch (no report
file was written at the time). Intermediate reasoning and tool
traces were not preserved. See `wiki/reports/README.md`.

---
Self-improvement for workflow (not kernels): Aug-Oct 2026 hunt.

### 1. Recent list (verified dates, primary first)

1. **MiMo-V2.6: Scaling RL Towards Self-Improvement** — Xiaomi LLM-Core, 21 Sep 2026. https://arxiv.org/abs/2609.mimo-scaling-reinforcement-learning — Single mixed RL run (code+visual+cyber, 1568×16/step) + groupwise agentic grading (GRS/GAR) ranking passing patches by quality.
2. **WikiSkill: Compiling Agent Experience into Persistent Knowledge** — Google Research, 27 Aug 2026. https://arxiv.org/abs/2608.27454 — Raw traces → persistent wiki (patterns+logs) → skill proposer + validation gate; wiki never rolled back.
3. **Self-Evolving Recommendation System** — Google/YouTube, RecSys'26. https://arxiv.org/html/2602.10226 — Fast offline (proxy) + slow online (north-star) dual-loop agents + Experiment Journal; production launches.
4. **Self-Evolving Skills via Surrogate-Guided Solve-and-Reproduce (reSolve)** — Liu et al., 10 Aug 2026. https://arxiv.org/abs/2608.28638 — Decouple solving from fresh-container reproduce; surrogate verifier + beam search; cheap-model skills beat human 60.1→74.9%.
5. **SkillGLoW: Procedural-Family Consolidation** — Yan et al., 2 Sep 2026. https://arxiv.org/abs/2609.02217 — Local skills → de-instantiated global priors per procedure family + commit gate; 3.6× compact, +17.2pts.
6. **Designer-RSI: Evolving Procedural Memory from User Traffic** — Du et al., 18 Sep 2026. https://arxiv.org/abs/2609.22086 — Frozen model + widen (new subtask) / deepen (revise on own succ/fail) + matched-replay gate; 76→139 skills, 72.7→99.3%.
7. **SkillPivot: Deviation-Guided Self-Evolution** — Sep 24 2026. https://arxiv.org/abs/2609.29154 — Locate prefix→suffix deviation point, teacher-continues-from-prefix, contrast suffixes to minimal delta + regression verifier.
8. **When AI builds itself (RSI)** — Anthropic Institute, 2026. https://www.anthropic.com/institute/recursive-self-improvement — Field report: 80% code by Claude (May 26), 3×→52× kernel speedups, weak-to-strong agents close 97% gap; steering-judgment data.

Hype flag: MiMo/Anthropic numbers are self-reported, single-run, unreplicated; skill papers use small held-out sets.

### 2. Top-3 mechanisms (implementable)

**WikiSkill:** Artifact: `wiki/patterns/*.md + logs.md + skill-impact.md` + `skills/*.md`. When: after each kernel task batch. From: immutable traces. Validate: candidate skill → held-out validation split; accept iff score > best; else revert skill, keep wiki.
**Designer-RSI:** Artifact: procedural skill edits (widen/deepen). When: on recurring uncovered subtask or failed exec. From: succ+fail pairs for same procedure. Validate: matched replay — must fix target failure, zero regressions on prior successes; else discard.
**reSolve:** Artifact: self-contained skill package. When: per-task skill proposal. From: solver trajectory. Validate: independent reproduce in fresh container + surrogate verifier (KLD/perf/retention proxy, no hidden tests) + beam search keep-best.

### 3. Integration proposals (ranked)

**A. Rejected-log miner → skill PRs (high value/low risk).** Change: nightly subagent clusters `doc/kvarn-4090.md` rejected approaches into `patterns/` + proposes one skill edit with twin-test gate. Cost: ~6-10h. Breaks: bloated/generic skills. Measure: repeat-failure rate, skill acceptance %.
**B. Twin-test auto-gen from diffs (med effort, high value).** Change: on each kernel diff, generate KLD + perf + retention twin via reSolve reproduce protocol; must pass warmed-box hot-cache anchors before skill commit. Cost: ~12-20h. Breaks: flaky gates. Measure: escaped regressions, gate runtime.
**C. Failure-taxonomy checklist injection (low effort).** Change: ETGPO/SkillGLoW-style: top-3 error clusters auto-injected into dispatch briefs as checklist. Cost: ~4-6h. Breaks: prompt bloat. Measure: first-pass pass rate per brief version.

### 4. NOT to adopt

1. **MiMo-style full RL / weight updates / Mendel Gödel self-rewriting:** needs $0.9-2.6M, 750k trajectories, 23k sandboxes; destroys bit-exact determinism and hot-cache comparability.
2. **LLM-judge-only / co-evolving critic (CAFE-style) without verifiers:** unverifiable graphics-design-like feedback drifts; our KLD/bit-exact gates must remain code, not voted prose. Also skip open-ended multi-agent debate for review — cost/latency exceeds single-validator twin suite benefit at our scale.
