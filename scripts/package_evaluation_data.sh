#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-data/embryo/test_data}"
OUTPUT_DIR="${OUTPUT_DIR:-release_assets}"
OUTPUT_FILE="${OUTPUT_FILE:-embryo-test-data.zip}"

if [ ! -d "$SOURCE_DIR/bad" ] || [ ! -d "$SOURCE_DIR/good" ]; then
  echo "Expected evaluation data at $SOURCE_DIR/{bad,good}"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 - "$SOURCE_DIR" "$OUTPUT_DIR/$OUTPUT_FILE" <<'PY'
import sys
import zipfile
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
root = source.parent

with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(root))

print(f"Created {output}")
PY
