# Task 1 report — `deploy.sh` ships `state.json` every time

Status: **DONE_WITH_CONCERNS** (the concern is about the brief's verification block, not the change)
Commit: `5134738b` — `bench: deploy.sh ships state.json on every run, reports updated/unchanged`

## What I implemented

In `bench/windows/deploy.sh`, the conditional install block (lines 23–27 before) was replaced with an
unconditional install plus a before/after sha256 comparison:

```bash
# The active regime lives in state.json; run_pass.ps1 refuses to start without it. The repo is the
# source of truth: a regime change is recorded here and deployed, never edited on the box. Shipping
# only when the destination lacks the file leaves the box on an old regime while
# bench/verify_task.py --pre validates the new one, so it ships on every deploy.
state_sha_before="$(sha256sum "$DEST/state.json" 2>/dev/null | cut -d' ' -f1 || true)"
install -m 0666 "$HERE/state.json" "$DEST/state.json"
if [ "$state_sha_before" = "$(sha256sum "$DEST/state.json" | cut -d' ' -f1)" ]; then
  echo "state.json: unchanged"
else
  echo "state.json: updated"
fi
```

and the closing line became:

```bash
sha256sum "$DEST"/*.ps1 "$DEST"/perf.py "$DEST"/state.json
```

Requirement mapping:

- Always `install -m 0666 "$HERE/state.json" "$DEST/state.json"` — unconditional, as written.
- `state.json: updated` on byte change, `state.json: unchanged` otherwise — sha256 before/after (the
  brief allowed either `cmp -s` against a temp copy or a hash comparison); missing destination file
  hashes to `""`, so the first deploy reports `updated`.
- Old comment replaced by the new rule: repo is the source of truth, a regime change is recorded and
  deployed, never edited on the box. The comment also states *why* the old guard was wrong (box keeps
  an old regime while `--pre` validates the new one).
- `state.json` appended to the closing `sha256sum` line.

Nothing else changed: no new files, no other script touched, `set -euo pipefail` and the existing
style kept, `bash -n` clean. No deploy was run against `_agent/v165`, no process started or killed,
no GPU work.

## Verification

Exactly the brief's block, run from the repo root with `DEST=/tmp/opencode/deploy-test`
(`set -o pipefail` on; only the extra `grep`s/`restore` steps are additions, marked below):

```
$ rm -rf /tmp/opencode/deploy-test
$ DEST=/tmp/opencode/deploy-test bench/windows/deploy.sh
state.json: updated
deployed to /tmp/opencode/deploy-test
6d99e40fe337859b54076ab35cb6eac4bafac62612dc311694a2114e71b37e99  /tmp/opencode/deploy-test/run_build.ps1
8dfd8b0a45772393219b6754131d9c8434e93f2d80ce8c8e181cad13aed4ce17  /tmp/opencode/deploy-test/run_pass.ps1
910f05a2821eb0e57b764bbcc13f369041e7479ad913dc5d13bed17cfdb47103  /tmp/opencode/deploy-test/watch.ps1
28d9afdd8a06da048b88330d03cddf6fe8bc073c5fb16ab151fbd9bde0fac1e2  /tmp/opencode/deploy-test/perf.py
7913608f838f5a6511d024abb36b5e4d098c07916ef6a4de53f291e8719d0dd4  /tmp/opencode/deploy-test/state.json

$ DEST=/tmp/opencode/deploy-test bench/windows/deploy.sh
state.json: unchanged
deployed to /tmp/opencode/deploy-test
... (same four script hashes)
7913608f838f5a6511d024abb36b5e4d098c07916ef6a4de53f291e8719d0dd4  /tmp/opencode/deploy-test/state.json

$ sed -i 's/"split": 376/"split": 372/' bench/windows/state.json
$ DEST=/tmp/opencode/deploy-test bench/windows/deploy.sh
state.json: updated
deployed to /tmp/opencode/deploy-test
... (same four script hashes)
eb50be881ad6cb42979fbd385d4811f052f09c437a777b2efdf86e5d50eb82c1  /tmp/opencode/deploy-test/state.json

$ grep -m1 '"split"' /tmp/opencode/deploy-test/state.json          # added: prove the dest bytes changed
    "split": 372,

$ git checkout -- bench/windows/state.json
$ DEST=/tmp/opencode/deploy-test bench/windows/deploy.sh
state.json: updated
deployed to /tmp/opencode/deploy-test
... (same four script hashes)
2dac005d19a2758abc6c28baf96c1eeaea4ada5b1e974abf29a228553406b166  /tmp/opencode/deploy-test/state.json

$ grep -m1 '"split"' /tmp/opencode/deploy-test/state.json          # added: the revert is deployed too
    "split": 376,
```

Repo `state.json` is byte-identical to `HEAD` (see "Concern 1" for the extra step that was needed):

```
$ sha256sum bench/windows/state.json
7913608f838f5a6511d024abb36b5e4d098c07916ef6a4de53f291e8719d0dd4  bench/windows/state.json
$ git show HEAD:bench/windows/state.json | sha256sum
7913608f838f5a6511d024abb36b5e4d098c07916ef6a4de53f291e8719d0dd4  -
$ git diff --stat -- bench/windows/state.json      # no output
$ git status --short
 M bench/windows/deploy.sh
?? wiki/plans/v165-critique-fixes.md     (pre-existing, untouched, not committed)
$ bash -n bench/windows/deploy.sh                  # no output
```

Both paths are exercised: `updated` on a fresh destination, `unchanged` on a repeat, `updated` on a
repo regime change with the deployed copy actually holding `split: 372`, `updated` again on the
revert with the deployed copy back at `376`.

## Files changed

- `bench/windows/deploy.sh` — the only file in the commit (+11 / −5).
- `.superpowers/sdd/v165-critique-fixes/task-1-report.md` — this report (untracked, as reports are).

## Self-review

- Every brief requirement met, verbatim where the brief gave exact text (`install -m 0666 …`,
  `state.json: updated`, `state.json: unchanged`); nothing added beyond them.
- Error handling: `sha256sum` on a missing destination is `2>/dev/null | cut -d' ' -f1 || true`, so
  `set -e` does not abort the first deploy. `sha256sum` is already a dependency of the script's last
  line, so no new tool requirement.
- Output order: `state.json: …` prints during the install block, before `deployed to $DEST`, matching
  the brief's transcripts.
- Not overbuilt: no `--dry-run`, no diff printing, no extra state.

## Concerns

1. **The brief's `git checkout -- bench/windows/state.json` silently changes the worktree bytes** in
   this repo. `core.autocrlf=true` is set and the blob in `HEAD` is LF, so `git checkout --` writes
   the file as CRLF: `git diff` stays clean (git normalizes), but the bytes change from sha256
   `7913608f…` (LF) to `2dac005d…` (CRLF). Since `deploy.sh` ships *worktree* bytes, a re-run of the
   verification block leaves the box holding a CRLF `state.json` while `git show HEAD:…` is LF — no
   functional effect (`verify_task.py` parses JSON, it does not compare hashes to HEAD), but the
   lane's "byte-identical to HEAD" criterion is not what `git diff --stat` measures.
   I restored the original bytes and cleared the resulting stat-dirty index entry:

   ```
   git show HEAD:bench/windows/state.json > bench/windows/state.json   # back to 7913608f…
   git add -- bench/windows/state.json                                 # identical blob, nothing staged
   ```

   Worth knowing for any later task that reverts a tracked file in `bench/windows/` with
   `git checkout --`. (Persisted to Engram as `bench/windows/deploy-state-json`.)

2. **Nothing was deployed to the Windows box.** The brief's verification is offline (`DEST=/tmp/…`)
   and the constraints forbid touching the host unasked, so `_agent/v165/state.json` still holds
   whatever it held. A real `bench/windows/deploy.sh` run is needed before the next measurement pass
   for the box to pick up this behaviour (and to report `updated` for the first time there).
3. Minor: a *stale but identical* destination still reports `unchanged`, which is the wanted signal;
   a destination whose `state.json` was hand-edited on the box is now silently replaced by the repo
   copy — that is the intended rule, but it means an on-box regime edit is lost without a diff being
   printed. The brief asked for exactly this; flagging only so nobody is surprised later.
