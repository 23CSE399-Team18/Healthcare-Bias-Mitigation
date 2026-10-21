"""Download INSPIRE v1.4.2 (perioperative dataset, Seoul National University Hospital) from PhysioNet.

INSPIRE is restricted data, so this runs under your own PhysioNet account:
  1. Log in at https://physionet.org and open https://physionet.org/content/inspire/1.4.2/
  2. Under "Files", sign the "Korea Credentialed Health Data Agreement" (needs a credentialed
     PhysioNet account and the CITI "Data or Specimens Only Research" course).
  3. Run:  python download_inspire.py
     It asks for your PhysioNet username and password (nothing is stored), downloads every
     table into D:/physionet.org/files/inspire/1.4.2, resumes interrupted files, and checks
     every file against the project's SHA256SUMS.txt.

Options:
  --list             show the files and sizes, download nothing
  --only PATTERN...  only files matching these patterns, e.g. --only operations.csv* labs.csv*
  --skip PATTERN...  leave out files matching these patterns, e.g. --skip vitals.csv*
"""
import argparse
import base64
import fnmatch
import getpass
import hashlib
import os
import re
import sys
import urllib.error
import urllib.request
from urllib.parse import unquote

BASE = "https://physionet.org/files/inspire/1.4.2/"
OUT = "D:/physionet.org/files/inspire/1.4.2"
CHUNK = 1 << 20
SUMS = "SHA256SUMS.txt"


def make_get(auth):
    def get(url, headers=None):
        req = urllib.request.Request(url, headers={"User-Agent": "physionet-downloader/1.0", **(headers or {})})
        if auth:
            req.add_header("Authorization", "Basic " + auth)
        return urllib.request.urlopen(req, timeout=120)
    return get


def walk(get, base, prefix, skip):
    """Yield (relative path, size in bytes) for every file in the PhysioNet index under base+prefix."""
    html = get(base + prefix).read().decode("utf-8", "replace")
    for href, rest in re.findall(r'<a href="([^"]+)">[^<]*</a>([^\n]*)', html):
        if href.startswith(("../", "/", "?", "#", "http")):
            continue
        rel = prefix + unquote(href)
        if skip(rel):
            continue
        if rel.endswith("/"):
            yield from walk(get, base, rel, skip)
        else:
            size = rest.split()[-1] if rest.split() else ""
            yield rel, int(size) if size.isdigit() else 0


def download(get, url, dest, size):
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    try:
        r = get(url, {"Range": f"bytes={have}-"} if have else None)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:                 # .part already holds the whole file
            os.replace(part, dest)
            return
        raise
    append = have and r.status == 206
    done = have if append else 0
    with r, open(part, "ab" if append else "wb") as f:
        while chunk := r.read(CHUNK):
            f.write(chunk)
            done += len(chunk)
            if size:
                print(f"\r    {done / 1e6:,.1f} / {size / 1e6:,.1f} MB", end="", flush=True)
    print()
    os.replace(part, dest)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--base", default=BASE, help="PhysioNet files URL of the project")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="PATTERN")
    ap.add_argument("--skip", nargs="+", metavar="PATTERN", default=[])
    ap.add_argument("--no-login", action="store_true", help="for open-access projects")
    a = ap.parse_args()

    auth = None
    if not a.no_login:
        user = input("PhysioNet username: ").strip()
        auth = base64.b64encode(f"{user}:{getpass.getpass('PhysioNet password: ')}".encode()).decode()
    get = make_get(auth)

    def matches(rel, patterns):
        return any(fnmatch.fnmatch(os.path.basename(rel), p) or fnmatch.fnmatch(rel, p) for p in patterns)

    def skip(rel):
        if rel.endswith("/") or os.path.basename(rel) == SUMS:
            return False
        return matches(rel, a.skip) or bool(a.only) and not matches(rel, a.only)

    try:
        files = list(walk(get, a.base, "", skip))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            sys.exit(f"PhysioNet refused access (HTTP {e.code}). Check your username and password, and that "
                     f"you signed the data use agreement on {a.base.replace('/files/', '/content/')}")
        raise

    for rel, size in files:
        print(f"{size / 1e6:12,.1f} MB  {rel}")
    print(f"{len(files)} files, {sum(s for _, s in files) / 1e9:,.2f} GB -> {a.out}")
    if a.list:
        return

    files.sort(key=lambda f: f[0] != SUMS)         # checksums first
    sums, bad = {}, []
    for i, (rel, size) in enumerate(files, 1):
        dest = os.path.join(a.out, *rel.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tag = f"[{i}/{len(files)}] {rel}"
        if rel in sums and os.path.exists(dest) and sha256(dest) == sums[rel]:
            print(f"{tag}  already downloaded, OK")
            continue
        print(tag)
        download(get, a.base + rel, dest, size)
        if rel == SUMS:
            for line in open(dest, encoding="utf-8"):
                if line.strip():
                    h, name = line.split(maxsplit=1)
                    sums[name.strip().lstrip("*")] = h.lower()
        elif rel not in sums:
            print("    OK (no checksum listed)")
        elif sha256(dest) == sums[rel]:
            print("    OK")
        else:
            os.replace(dest, dest + ".bad")
            bad.append(rel)
            print(f"    checksum MISMATCH, saved as {rel}.bad")

    if bad:
        sys.exit(f"{len(bad)} file(s) failed the checksum; run the script again to re-download them")
    print(f"done: {len(files)} files in {a.out}, all checksums OK")


if __name__ == "__main__":
    main()
