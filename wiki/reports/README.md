# Subagent reports (WikiSkill-lite raw layer)

Every research subagent writes its FULL report here and returns only
a short summary in chat. Context is expensive; files are cheap.
A report nobody can find later is the same as no report.

## Convention

- Path: `wiki/reports/YYYY-MM-DD-<topic>.md` (date = dispatch date).
- Dispatch prompt MUST contain (verbatim block):

  > Write your full report to `<path>` (findings, evidence with
  > file:line refs, dead ends, open questions). Return in chat ONLY:
  > status (DONE/BLOCKED), ≤15-line summary, and anything blocking
  > follow-up work. Do not paste the full report in chat.

- After the report lands: `ctx_index` it (source label
  `report-<topic>`) so later sessions retrieve it via `ctx_search`
  instead of re-running the research. Knowledge base survives
  compaction; chat doesn't.
- Backfill rule: none. Lost reports stay lost (lesson that created
  this file); do not reconstruct from memory.

## Why not just longer chat summaries

Chat summaries are lossy by design (the coordinator compresses for
the next step) and die with the session. Files keep evidence
(file:line refs, rejected alternatives, exact numbers) that future
work needs and summaries drop. The coordinator's context stays lean
while the knowledge compounds -- thorough-on-disk, terse-in-chat.
