"""Đóng gói bài nộp:  python package.py 2012345 2012346
Tạo file  <MSSV1-MSSV2>.zip  nằm cạnh thư mục bài làm (đúng định dạng đề yêu cầu).
- Bỏ __pycache__, .DS_Store, .venv.
- File lớn hơn --max-mb (mặc định 20MB) bị bỏ ra, và được liệt kê để m upload Drive rồi dán link vào README.
- Mặc định không đưa models/*.joblib vào zip (thường rất nặng); thêm --with-models nếu muốn.
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {"__pycache__", ".venv", ".git", ".ipynb_checkpoints"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mssv", nargs="+", help="danh sách MSSV của nhóm")
    ap.add_argument("--max-mb", type=float, default=20)
    ap.add_argument("--with-models", action="store_true")
    args = ap.parse_args()

    stem = "-".join(args.mssv)
    out = ROOT.parent / f"{stem}.zip"
    skipped_big, n = [], 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            if p.is_dir() or any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
                continue
            if p.name == ".DS_Store" or p.suffix == ".pyc":
                continue
            if p.suffix == ".joblib" and not args.with_models:
                continue
            mb = p.stat().st_size / 1e6
            if mb > args.max_mb:
                skipped_big.append((p.relative_to(ROOT), mb))
                continue
            z.write(p, Path(stem) / p.relative_to(ROOT))
            n += 1
    print(f"Đã tạo {out}  ({n} file, {out.stat().st_size / 1e6:.1f} MB)")
    if skipped_big:
        print("\nCác file lớn KHÔNG đưa vào zip - hãy upload (Drive/GitHub) và dán link công khai vào README.md:")
        for p, mb in skipped_big:
            print(f"  - {p}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
