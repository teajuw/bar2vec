#!/bin/bash
# One hill-climb round: record the change against the reference variant, run every song, print the table.
# Usage: scripts/hx_round.sh vN <reference-variant> "one-line change" "why (goes in change.md)"
set -e
V=$1; REF=$2; DESC=$3; WHY=$4
F=.claude/hillclimb/song-cuts
mkdir -p $F/$V
{ diff -u $F/$REF/cuts.json cuts.json || true; diff -u $F/$REF/cuts.py scripts/cuts.py || true; } > $F/$V/change.patch
printf '# %s: %s\n\n%s\n' "$V" "$DESC" "$WHY" > $F/$V/change.md
uv run scripts/hx_run.py --variant $V --description "$DESC" 2>&1 | tail -1
uv run scripts/hx_report.py --ref $REF
