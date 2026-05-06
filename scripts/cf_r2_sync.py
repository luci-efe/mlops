"""
cf_r2_sync.py — Sync DVC cache and MLflow artifacts to Cloudflare R2 using
the Cloudflare REST API + a Cloudflare API token (cfat).

Why: R2's S3-compatible endpoint requires AWS SigV4 signing, which needs an
Access Key ID + Secret Access Key. boto3 / dvc[s3] / mlflow's S3 artifact
store only speak SigV4. A Cloudflare API token (cfat_...) authenticates only
against api.cloudflare.com — not the S3 endpoint. Until we have proper R2
S3 keys, this script bridges the gap by uploading via the management REST
API (PUT/GET on /accounts/{account_id}/r2/buckets/{bucket}/objects/{key}).

Usage:
  CF_API_TOKEN=cfat_... python scripts/cf_r2_sync.py push
  CF_API_TOKEN=cfat_... python scripts/cf_r2_sync.py pull
  CF_API_TOKEN=cfat_... python scripts/cf_r2_sync.py list

Endpoints:
  PUT    /accounts/{acct}/r2/buckets/{bucket}/objects/{key}
  GET    /accounts/{acct}/r2/buckets/{bucket}/objects/{key}
  GET    /accounts/{acct}/r2/buckets/{bucket}/objects?per_page=1000
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ACCOUNT_ID = "48f381bf59212dbd98d2b424ba4b9a04"
BUCKET = "luci-mlops-fraud"
API_BASE = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/r2/buckets/{BUCKET}"

LOCAL_ROOTS = {
    "dvc-store": Path(".dvc/cache"),
    "mlflow":    Path("mlruns"),
}


def auth_header() -> dict:
    token = os.environ.get("CF_API_TOKEN")
    if not token:
        sys.exit("ERROR: CF_API_TOKEN env var not set")
    return {"Authorization": f"Bearer {token}"}


def put_one(local: Path, key: str) -> tuple[str, int, str]:
    url = f"{API_BASE}/objects/{key}"
    with local.open("rb") as f:
        r = requests.put(url, data=f, headers=auth_header(), timeout=120)
    return key, r.status_code, "ok" if r.ok else r.text[:200]


def push() -> None:
    files: list[tuple[Path, str]] = []
    for prefix, root in LOCAL_ROOTS.items():
        if not root.exists():
            print(f"[skip] {root} does not exist")
            continue
        for p in root.rglob("*"):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                files.append((p, f"{prefix}/{rel}"))

    print(f"Uploading {len(files)} files to s3://{BUCKET}/ via Cloudflare REST API …")

    ok = err = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(put_one, p, k): (p, k) for p, k in files}
        for fut in as_completed(futures):
            key, status, msg = fut.result()
            if status < 300:
                ok += 1
            else:
                err += 1
                print(f"  FAIL [{status}] {key}: {msg}")
    print(f"Done. ok={ok}  err={err}")


def list_objects() -> None:
    url = f"{API_BASE}/objects?per_page=1000"
    r = requests.get(url, headers=auth_header(), timeout=30)
    r.raise_for_status()
    objs = r.json()["result"]
    print(f"Bucket {BUCKET}: {len(objs)} objects")
    for o in objs[:50]:
        print(f"  {o['size']:>10}  {o['key']}")
    if len(objs) > 50:
        print(f"  ... and {len(objs) - 50} more")


def pull() -> None:
    url = f"{API_BASE}/objects?per_page=1000"
    r = requests.get(url, headers=auth_header(), timeout=30)
    r.raise_for_status()
    objs = r.json()["result"]
    for o in objs:
        key = o["key"]
        # Map prefix back to local path
        if "/" not in key:
            continue
        prefix, _, rel = key.partition("/")
        if prefix not in LOCAL_ROOTS:
            continue
        local = LOCAL_ROOTS[prefix] / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        gr = requests.get(f"{API_BASE}/objects/{key}", headers=auth_header(), timeout=120)
        gr.raise_for_status()
        local.write_bytes(gr.content)
        print(f"  pulled {key} → {local}")
    print("Done.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    {"push": push, "pull": pull, "list": list_objects}[cmd]()
