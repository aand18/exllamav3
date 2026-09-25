# Fork workflow (agents: follow this)

This checkout works on a fork with WIP branches. Rules:

- Never commit to `master`. `master` is a pristine mirror of upstream
  (`origin` = turboderp-org/exllamav3, `master` tracks `origin/master`).
- Work only on `feat/<issue>-<slug>`, `fix/<issue>-<slug>`, `wip/<topic>`.
- Before starting: `git fetch origin && git rebase origin/master`.
- Rebase, never merge `master` into feature branches.
- Push feature branches to the `fork` remote (aand18/exllamav3), never to `origin`.
- Document each branch with a Draft PR `branch -> master` on the fork with:
  goal, upstream issue link (if any), non-goals, current status, test plan.
- Keep the `BRANCHES.md` table current: branch | upstream issue | status | draft PR.
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
