"""Tải PDF các văn bản quy chế theo data/corpus/sources.csv.

PDF không đưa vào repo (xem .gitignore); chạy script này để tải lại:
    python -m rag.corpus.download
"""
import argparse
import csv
import urllib.request
from pathlib import Path

SOURCES = Path("data/corpus/sources.csv")
PDF_DIR = Path("data/pdfs")


def load_sources(path=SOURCES):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="tải lại cả file đã có")
    args = ap.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for src in load_sources():
        if not src["url"]:
            print(f"[bỏ qua] {src['doc_id']}: chưa có link")
            continue
        out = PDF_DIR / f"{src['doc_id']}.pdf"
        if out.exists() and not args.force:
            print(f"[đã có] {out}")
            continue
        req = urllib.request.Request(src["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            out.write_bytes(resp.read())
        print(f"[tải xong] {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
