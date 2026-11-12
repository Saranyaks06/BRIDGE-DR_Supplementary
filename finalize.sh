#!/usr/bin/env bash
# Regenerate the report, the supplementary document and the figures from whatever results exist,
# then commit locally. Usage: bash scripts/finalize.sh ["commit message"]
set -e
cd "$(dirname "$0")/.."
python scripts/export_report_data.py
python scripts/make_figures.py
python scripts/fill_report.py
python scripts/make_supplementary.py
git add -A
git commit -q -m "${1:-Regenerate report and supplementary from latest results}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "nothing to commit"
git log --oneline -1
