#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:-/scratch/sr7849/KV_Cache/riskkv_pipeline}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$DST/experiments/final_scaleup/code_backups/rest_pipeline_$STAMP"

if [[ ! -d "$DST" ]]; then
  echo "Destination repo missing: $DST" >&2
  exit 2
fi

mkdir -p "$BACKUP"

cd "$SRC"
while IFS= read -r rel; do
  [[ "$rel" == "apply_to_repo.sh" ]] && continue
  [[ "$rel" == README_RISKKV_REST.md ]] && continue
  src="$SRC/$rel"
  dst="$DST/$rel"
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]]; then
    mkdir -p "$BACKUP/$(dirname "$rel")"
    cp -a "$dst" "$BACKUP/$rel"
  fi
  cp -a "$src" "$dst"
  echo "INSTALLED $rel"
done < <(find riskkv scripts -type f ! -path '*/__pycache__/*' | sort)

if [[ -f "$DST/requirements_rest.txt" ]]; then
  cp -a "$DST/requirements_rest.txt" "$BACKUP/requirements_rest.txt"
fi
cp -a "$SRC/requirements_rest.txt" "$DST/requirements_rest.txt"
cp -a "$SRC/README_RISKKV_REST.md" "$DST/README_RISKKV_REST.md"

echo
echo "Installed into: $DST"
echo "Backup of replaced files: $BACKUP"
echo "Next:"
echo "  cd $DST"
echo "  export PYTHONPATH=\$PWD:\${PYTHONPATH:-}"
echo "  /home/sr7849/.conda/envs/riskkv/bin/pip install -r requirements_rest.txt"
echo "  /home/sr7849/.conda/envs/riskkv/bin/python scripts/final_scaleup/scripts/35_smoke_test_rest.py"
