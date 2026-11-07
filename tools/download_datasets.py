#!/usr/bin/env python3
"""One-click download of the project datasets from Google Drive.

Reads tools/datasets_manifest.json, downloads every file (or its split parts)
by Drive file ID, verifies SHA-256, rebuilds split files by concatenation and
writes them under <repo>/<local_root>. Stdlib only; `--gdown` is optional.

    python tools/download_datasets.py                 # everything
    python tools/download_datasets.py --only mlp_balanced
    python tools/download_datasets.py --list
"""
import argparse
import hashlib
import http.cookiejar
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent
DRIVE_URL = "https://drive.google.com/uc?export=download&id={}"
CHUNK = 1 << 20
SIGNIN_MSG = ("Google returned a sign-in page: the Drive folder is not shared with you / "
              "not public; ask the owner to share the Drive folder, or use --gdown")


class DownloadError(Exception):
    pass


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def human(n):
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"


_opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
_opener.addheaders = [("User-Agent", "Mozilla/5.0 (download_datasets.py)")]


def _confirm_url(html, resp_url):
    """Build the follow-up URL for Drive's 'can't scan for viruses' interstitial."""
    form = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"', html)
    if form:  # new style: form with hidden id/export/confirm/uuid inputs
        fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', html))
        return urllib.parse.unquote(form.group(1).replace("&amp;", "&")) + "?" + \
            urllib.parse.urlencode(fields)
    m = re.search(r'confirm=([0-9A-Za-z_\-]+)', html)  # old style: confirm token in a link
    if m:
        sep = "&" if "?" in resp_url else "?"
        uuid = re.search(r'uuid=([0-9A-Za-z_\-]+)', html)
        return resp_url + f"{sep}confirm={m.group(1)}" + (f"&uuid={uuid.group(1)}" if uuid else "")
    return None


def fetch(url, dest):
    """Stream `url` to `dest` (via dest.partial), handling Drive interstitials."""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".partial")
    for _ in range(3):  # original request + up to two confirm hops
        resp = _opener.open(url, timeout=60)
        final_url = resp.geturl()
        ctype = resp.headers.get("Content-Type", "")
        if "accounts.google.com" in final_url:
            raise DownloadError(SIGNIN_MSG)
        if ctype.startswith("text/html"):
            html = resp.read().decode("utf-8", "replace")
            if "accounts.google.com" in html and "download-form" not in html:
                raise DownloadError(SIGNIN_MSG)
            nxt = _confirm_url(html, final_url)
            if not nxt:
                raise DownloadError(f"unexpected HTML from Drive ({final_url})")
            url = nxt
            continue
        with open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, CHUNK)
        os.replace(tmp, dest)
        return dest
    raise DownloadError("Drive kept returning an interstitial page")


def fetch_by_id(file_id, dest, use_gdown=False, local_dir=None, name=None):
    if local_dir:  # debug: take the file from a local directory instead of Drive
        src = Path(local_dir) / (name or Path(dest).name)
        if not src.is_file():
            raise DownloadError(f"not in --from-local-parts dir: {src}")
        tmp = Path(dest).with_name(Path(dest).name + ".partial")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        return dest
    if use_gdown:
        import gdown
        tmp = str(dest) + ".partial"
        if not gdown.download(id=file_id, output=tmp, quiet=False):
            raise DownloadError(f"gdown failed for id {file_id}")
        os.replace(tmp, dest)
        return dest
    return fetch(DRIVE_URL.format(file_id), dest)


def process(entry, root, args):
    """Return (status, detail, bytes_downloaded) for one manifest entry."""
    dest = root / entry["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and sha256_of(dest) == entry["sha256"]:
        return "OK", "already present", 0
    got = 0
    if "parts" not in entry:
        fetch_by_id(entry["file_id"], dest, args.gdown, args.from_local_parts)
        got = dest.stat().st_size
    else:
        part_paths = []
        for p in entry["parts"]:
            pp = dest.parent / p["name"]
            if not (pp.is_file() and sha256_of(pp) == p["sha256"]):
                print(f"    part {p['name']}", flush=True)
                fetch_by_id(p["file_id"], pp, args.gdown, args.from_local_parts, p["name"])
                got += pp.stat().st_size
                if sha256_of(pp) != p["sha256"]:
                    pp.unlink()
                    return "FAIL", f"sha256 mismatch in {p['name']}", got
            part_paths.append(pp)
        tmp = dest.with_name(dest.name + ".partial")
        with open(tmp, "wb") as out:
            for pp in part_paths:
                with open(pp, "rb") as f:
                    shutil.copyfileobj(f, out, CHUNK)
        os.replace(tmp, dest)
    if sha256_of(dest) != entry["sha256"]:
        return "FAIL", "sha256 mismatch", got
    if "parts" in entry and not args.keep_parts:
        for p in entry["parts"]:
            (dest.parent / p["name"]).unlink(missing_ok=True)
    return "OK", "downloaded" if not args.from_local_parts else "rebuilt from local", got


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download project datasets from Google Drive.")
    ap.add_argument("--manifest", type=Path, default=TOOLS_DIR / "datasets_manifest.json")
    ap.add_argument("--root", type=Path, help="target dir (default: <repo>/<local_root>)")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="substring filter(s) on path")
    ap.add_argument("--list", action="store_true", help="print manifest table and exit")
    ap.add_argument("--keep-parts", action="store_true", help="keep split parts after rebuild")
    ap.add_argument("--gdown", action="store_true", help="use gdown instead of urllib")
    ap.add_argument("--from-local-parts", metavar="DIR", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    root = (args.root or REPO_DIR / manifest["local_root"]).resolve()
    files = [f for f in manifest["files"]
             if not args.only or any(s in f["path"] for s in args.only)]
    if not files:
        sys.exit(f"no manifest entries match --only {args.only}")
    width = max(len(f["path"]) for f in files)

    if args.list:
        print(f"Drive folder: {manifest['drive_folder_url']}\nRoot: {root}\n")
        print(f"{'path':<{width}}  {'size':>10}  parts  sha256")
        for f in files:
            print(f"{f['path']:<{width}}  {human(f['size']):>10}  {len(f.get('parts', [])):>5}"
                  f"  {f['sha256'][:16]}")
        print(f"\n{len(files)} files, {human(sum(f['size'] for f in files))} total")
        return 0
    if args.gdown:
        try:
            import gdown  # noqa: F401
        except ImportError:
            sys.exit("--gdown requested but gdown is not installed (pip install gdown)")

    print(f"Downloading {len(files)} files into {root}")
    results, total = [], 0
    for f in files:
        print(f"  {f['path']} ({human(f['size'])})", flush=True)
        try:
            status, detail, got = process(f, root, args)
        except DownloadError as e:
            status, detail, got = "FAIL", str(e), 0
            if SIGNIN_MSG in str(e):
                print(f"\nERROR: {e}\nFolder: {manifest['drive_folder_url']}", file=sys.stderr)
                results.append((f["path"], status, "sign-in required"))
                break
        except OSError as e:
            status, detail, got = "FAIL", f"{type(e).__name__}: {e}", 0
        total += got
        results.append((f["path"], status, detail))

    print(f"\n{'path':<{width}}  status  detail")
    for path, status, detail in results:
        print(f"{path:<{width}}  {status:<6}  {detail}")
    for f in files[len(results):]:
        print(f"{f['path']:<{width}}  {'SKIP':<6}  not attempted")
    n_ok = sum(s == "OK" for _, s, _ in results)
    print(f"\n{n_ok}/{len(files)} OK, {human(total)} transferred, root: {root}")
    return 0 if n_ok == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
