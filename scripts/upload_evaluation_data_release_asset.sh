#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-blaster3618/embryo-health-prediction}"
TAG="${TAG:-evaluation-data-v1}"
ASSET="${ASSET:-release_assets/embryo-test-data.zip}"
REPLACE_ASSET="${REPLACE_ASSET:-0}"

if [ ! -f "$ASSET" ]; then
  bash scripts/package_evaluation_data.sh
fi

echo "Uploading evaluation data asset:"
echo "  $ASSET"
echo

if command -v gh >/dev/null 2>&1; then
  if ! gh auth status >/dev/null 2>&1; then
    echo "Run 'gh auth login' before uploading release assets."
    exit 1
  fi

  clobber_flag=()
  if [ "$REPLACE_ASSET" = "1" ]; then
    clobber_flag=(--clobber)
  fi

  if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    gh release upload "$TAG" "$ASSET" --repo "$REPO" "${clobber_flag[@]}"
  else
    gh release create "$TAG" "$ASSET" \
      --repo "$REPO" \
      --title "Evaluation data v1" \
      --notes "Labelled test-data subset for the Streamlit research prototype evaluation tab."
  fi
  exit 0
fi

if [ -z "${GITHUB_TOKEN:-}" ]; then
  cat <<EOF
GitHub CLI is not installed, and GITHUB_TOKEN is not set.

Use either option:

1. Install GitHub CLI, then run:
   gh auth login
   bash scripts/upload_evaluation_data_release_asset.sh

2. Or create a GitHub token with repo/release permission, then run:
   export GITHUB_TOKEN=your_token_here
   bash scripts/upload_evaluation_data_release_asset.sh

This uploads a zip of data/embryo/test_data only. It does not upload the full
training or validation dataset.
EOF
  exit 1
fi

python3 - "$REPO" "$TAG" "$ASSET" "$REPLACE_ASSET" <<'PY'
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

repo, tag, asset_path, replace_asset = sys.argv[1:]
asset = Path(asset_path)
token = os.environ["GITHUB_TOKEN"]
api_root = f"https://api.github.com/repos/{repo}"
headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {token}",
    "X-GitHub-Api-Version": "2022-11-28",
}


def request(method, url, data=None, content_type="application/vnd.github+json"):
    body = None
    req_headers = dict(headers)
    if data is not None:
        if isinstance(data, (dict, list)):
            body = json.dumps(data).encode("utf-8")
        else:
            body = data
        req_headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw.decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        if exc.code == 404:
            return exc.code, None
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code} {raw}") from exc


status, release = request("GET", f"{api_root}/releases/tags/{urllib.parse.quote(tag)}")
if status == 404:
    _, release = request("POST", f"{api_root}/releases", {
        "tag_name": tag,
        "name": "Evaluation data v1",
        "body": "Labelled test-data subset for the Streamlit evaluation tab.",
    })

release_id = release["id"]
upload_url = release["upload_url"].split("{", 1)[0]
_, existing_assets = request("GET", f"{api_root}/releases/{release_id}/assets?per_page=100")
existing_by_name = {item["name"]: item["id"] for item in existing_assets}

if asset.name in existing_by_name:
    if replace_asset != "1":
        print(f"Skipping existing asset {asset.name}. Set REPLACE_ASSET=1 to replace it.")
        raise SystemExit(0)
    request("DELETE", f"{api_root}/releases/assets/{existing_by_name[asset.name]}")

print(f"Uploading {asset} ...")
request(
    "POST",
    f"{upload_url}?name={urllib.parse.quote(asset.name)}",
    data=asset.read_bytes(),
    content_type="application/octet-stream",
)
print("Upload complete.")
PY
