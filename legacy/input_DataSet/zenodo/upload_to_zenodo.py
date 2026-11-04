"""Create a Zenodo DRAFT with the three CSVs, README.md and SHA256SUMS.txt from this folder.

The draft is NOT published: open the printed link, check everything, then press
Publish yourself. Published Zenodo records and DOIs cannot be deleted.

Usage (PowerShell):
    $env:ZENODO_TOKEN = "<personal access token with deposit:write>"
    python upload_to_zenodo.py
    python upload_to_zenodo.py --sandbox   # test run on sandbox.zenodo.org
"""
import argparse
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["diabetes.csv", "heart.csv", "hypertension.csv", "README.md", "SHA256SUMS.txt"]


def check(resp):
    if not resp.ok:
        sys.exit(f"{resp.request.method} {resp.url} -> {resp.status_code}\n{resp.text}")
    return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set ZENODO_TOKEN first.")

    with open(os.path.join(HERE, "zenodo_metadata.json"), encoding="utf-8") as f:
        record = json.load(f)

    paths = [os.path.join(HERE, n) for n in FILES]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        sys.exit("Missing files:\n" + "\n".join(missing))

    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"

    draft = check(s.post(f"{base}/api/records", json=record)).json()
    rid = draft["id"]
    print(f"Created draft {rid}")

    check(s.post(f"{base}/api/records/{rid}/draft/files", json=[{"key": k} for k in FILES]))
    for path, key in zip(paths, FILES):
        print(f"Uploading {key} ({os.path.getsize(path) / 1e6:.1f} MB)")
        with open(path, "rb") as f:
            check(s.put(f"{base}/api/records/{rid}/draft/files/{key}/content",
                        data=f, headers={"Content-Type": "application/octet-stream"}))
        check(s.post(f"{base}/api/records/{rid}/draft/files/{key}/commit"))

    print(f"\nDraft ready (not published). Review and publish at:\n{base}/uploads/{rid}")


if __name__ == "__main__":
    main()
