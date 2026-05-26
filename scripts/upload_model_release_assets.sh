#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-blaster3618/embryo-health-prediction}"
TAG="${TAG:-model-weights-v1}"
REPLACE_ASSETS="${REPLACE_ASSETS:-0}"
ASSETS=(saved_models/*_best.pt)

if [ "${ASSETS[0]}" = "saved_models/*_best.pt" ]; then
  echo "No model weights found at saved_models/*_best.pt"
  exit 0
fi

echo "Uploading ${#ASSETS[@]} model files from saved_models/ only:"
printf '  %s\n' "${ASSETS[@]}"
echo

if command -v gh >/dev/null 2>&1; then
  if ! gh auth status >/dev/null 2>&1; then
    echo "Run 'gh auth login' before uploading release assets."
    exit 1
  fi

  clobber_flag=()
  if [ "$REPLACE_ASSETS" = "1" ]; then
    clobber_flag=(--clobber)
  fi

  if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    gh release upload "$TAG" "${ASSETS[@]}" --repo "$REPO" "${clobber_flag[@]}"
  else
    gh release create "$TAG" "${ASSETS[@]}" \
      --repo "$REPO" \
      --title "Model weights v1" \
      --notes "PyTorch model weights for the Streamlit deployment. The app downloads these lazily instead of cloning them through Git LFS."
  fi
  exit 0
fi

if [ -z "${GITHUB_TOKEN:-}" ]; then
  cat <<EOF
GitHub CLI is not installed, and GITHUB_TOKEN is not set.

Use either option:

1. Install GitHub CLI, then run:
   gh auth login
   bash scripts/upload_model_release_assets.sh

2. Or create a GitHub token with repo/release permission, then run:
   export GITHUB_TOKEN=your_token_here
   bash scripts/upload_model_release_assets.sh

This script uploads only saved_models/*_best.pt. It does not upload legacy
resnet18/*.pt or resnet50/*.pt files.
EOF
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for the GITHUB_TOKEN upload fallback."
  exit 1
fi

python3 - "$REPO" "$TAG" "$REPLACE_ASSETS" "${ASSETS[@]}" <<'PY'
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

repo, tag, replace_assets, *assets = sys.argv[1:]
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
        "name": "Model weights v1",
        "body": "PyTorch model weights for lazy Streamlit downloads.",
    })

release_id = release["id"]
upload_url = release["upload_url"].split("{", 1)[0]
_, existing_assets = request("GET", f"{api_root}/releases/{release_id}/assets?per_page=100")
existing_by_name = {asset["name"]: asset["id"] for asset in existing_assets}

for asset in assets:
    path = Path(asset)
    name = path.name
    if name in existing_by_name:
        if replace_assets != "1":
            print(f"Skipping existing asset {name}.")
            continue
        request("DELETE", f"{api_root}/releases/assets/{existing_by_name[name]}")

    url = f"{upload_url}?name={urllib.parse.quote(name)}"
    print(f"Uploading {asset} ...")
    data = path.read_bytes()
    request("POST", url, data=data, content_type="application/octet-stream")

print("Upload complete.")
PY
