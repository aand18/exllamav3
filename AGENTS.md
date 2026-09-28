# Fork workflow: canonical copy lives on fork-overview

Do not create or edit workflow docs on this branch. Read and follow the
canonical files instead (fetch first: clones have no local
fork-overview branch, so the bare `git show fork-overview:...` form
fails):

    git fetch origin fork-overview
    git show origin/fork-overview:AGENTS.md
    git show origin/fork-overview:BRANCHES.md

Status and row updates go directly to `fork-overview` (docs-only commits,
pushed to the `fork` remote).
