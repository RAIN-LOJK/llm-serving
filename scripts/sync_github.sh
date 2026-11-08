#!/usr/bin/env bash
# Push the current main to the GitHub read-only mirror.
#
# Development stays on origin (AtomGit). This script refuses anything that is not a
# fast-forward of an already-pushed origin/main, so GitHub can never get a commit that
# the primary repository does not have.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

branch="$(git symbolic-ref --short HEAD)"
if [ "$branch" != "main" ]; then
  echo "sync_github: not on main (on $branch)" >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "sync_github: worktree is not clean:" >&2
  git status --short >&2
  exit 1
fi

if [ "$(git config --get branch.main.remote || true)" != "origin" ]; then
  echo "sync_github: branch.main.remote is not origin; refusing to change the primary" >&2
  exit 1
fi

git fetch origin
if [ "$(git rev-parse main)" != "$(git rev-parse origin/main)" ]; then
  echo "sync_github: local main != origin/main; push origin first" >&2
  exit 1
fi

git fetch github
if git rev-parse --verify -q github/main >/dev/null; then
  if ! git merge-base --is-ancestor github/main main; then
    echo "sync_github: github/main is not an ancestor of main; refusing non-fast-forward" >&2
    echo "sync_github: reconcile on origin, then retry; do not force-push" >&2
    exit 1
  fi
elif git ls-remote --exit-code --heads github main >/dev/null 2>&1; then
  echo "sync_github: github has main but it was not fetched; run git fetch github" >&2
  exit 1
fi

git push github main
# Tags are not mirrored by default: the legacy tag is intentionally not published.
echo "sync_github: pushed $(git rev-parse --short HEAD) to github/main"
