#!/bin/bash
# Post-MCMC pipeline: after 03b BEACON MCMC completes, regenerate the
# downstream artifacts that depend on the lineage-corrected differential
# EDD output.
#
# Run from repo root:
#   ./scripts/run_post_mcmc.sh

set -euo pipefail

PY=/opt/anaconda3/envs/taskweaver/bin/python

echo "=== [1/5] Hallmark enrichment on corrected solid-tumour targets ==="
$PY scripts/08_hallmark_enrichment.py

echo "=== [2/5] ICB benchmark with corrected solid-tumour signature ==="
$PY scripts/05c_icb_benchmark_solid.py

echo "=== [3/5] Evidence integration with corrected differential EDD ==="
$PY scripts/07_integration.py \
    --diff-edd-path analysis/out/immune_context_solid/differential_edd_solid_hot_vs_cold.csv \
    --suffix _solid \
    --exclude-heme-lineages

echo "=== [4/5] Regenerate Fig 5 (ICB benchmark, both signatures) ==="
$PY - <<'PY'
import sys, importlib.util
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
spec = importlib.util.spec_from_file_location('gf', 'scripts/generate_figures.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.fig5_icb_benchmark()
PY

echo "=== [5/5] Regenerate Fig 6 (two versions: all-lineages and solid-only) ==="
$PY - <<'PY'
import sys, importlib.util
from pathlib import Path
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
spec = importlib.util.spec_from_file_location('gf', 'scripts/generate_figures.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.fig6_evidence_integration()  # default: all-lineages
m.fig6_evidence_integration(
    evidence_path=Path('analysis/out/integration/beacon_io_evidence_table_solid.csv'),
    out_suffix='_solid',
    title_suffix=' (Solid-tumour, lineage-corrected)',
)
PY

echo "=== DONE ==="
