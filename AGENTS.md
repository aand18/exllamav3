# Fork workflow (agents: follow this)

This checkout works on a fork with WIP branches. Rules:

- Never commit to `master`. `master` is a pristine mirror of upstream
  (`origin` = turboderp-org/exllamav3, `master` tracks `origin/master`).
- Work only on `feat/<issue>-<slug>`, `fix/<issue>-<slug>`, `wip/<topic>`.
- Start every new branch from `master`, never from another feature branch:
  `git checkout master && git pull` (fast-forward only — if it conflicts,
  stop, `master` was polluted), then `git checkout -b <name>`.
- Verify the base before the first commit: `git merge-base --is-ancestor
  master HEAD` must pass and `git rev-parse HEAD` must equal
  `git rev-parse master`. If HEAD already contains another branch's commits,
  delete the branch and start over — a contaminated base pollutes the PR
  diff and can't be independently merged.
- Before starting: `git fetch origin && git rebase origin/master`.
- Rebase, never merge `master` into feature branches.
- Push feature branches to the `fork` remote (aand18/exllamav3), never to `origin`.
- Integration branches (testing/building several features together):
  cut `wip/integration-<target>` from a fast-forward-pulled `master`;
  rebase each feature onto `master` first, then `git merge --no-ff` them in
  one at a time with the suite green after each. Never rebase the
  integration branch and never merge it anywhere (not into features,
  `master`, or upstream) — rebuild it from scratch when a feature updates,
  noting conflict resolutions so they can be replayed. Dependent (stacked)
  features are named `wip/<topic>-stacked-on-<base>` and noted in
  `BRANCHES.md`. Extension rebuilds happen in a scratch copy, never in the
  working checkout; `git status` must show no build artifacts before commit.
- Document each branch with a Draft PR `branch -> master` on the fork with:
  goal, upstream issue link (if any), non-goals, current status, test plan.
- Keep the `BRANCHES.md` table current: branch | upstream issue | status | draft PR.
  Each table row gets a detail section below the table (goal, non-goals,
  status, test plan, verify command, key commits, blocked-on, history notes).
- Keep `master` pullable: no extra commits, no docs edits on `master`.
- `fork-overview` is the fork's GitHub default branch and human entry point:
  static README FORK header plus canonical copies of this file and
  `BRANCHES.md`. The header stays static and links out (branch table lives
  in `BRANCHES.md`) so upstream README changes stay mergeable; sync it with
  `git rebase origin/master` and keep the header on conflicts.
- Feature branches follow `feat/<issue>-<slug>`, `fix/<issue>-<slug>`,
  `wip/<topic>` naming (this branch was renamed `kvarn` ->
  `wip/kvarn-cache` to comply).
- This repo's standing rules: CPU-only test env here means no CUDA/Triton/ext
  builds unless stated; always use a venv, never touch system Python/config;
  atomic commits with the why in the message; never change git config
  (global or local) without being asked.
