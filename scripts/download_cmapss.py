"""
C-MAPSS Dataset Downloader
===========================
Downloads the NASA C-MAPSS turbofan run-to-failure dataset.

Tries three methods in order:
  1. Direct per-file download from a public GitHub mirror
  2. Kaggle CLI (if kaggle package + credentials are configured)
  3. Manual download instructions

Usage:
    python scripts/download_cmapss.py
"""

import os
import sys
import urllib.request
import zipfile

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

DEST_DIR = os.path.join("data", "cmapss")

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

EXPECTED_FILES = []
for s in SUBSETS:
    EXPECTED_FILES += [f"train_{s}.txt", f"test_{s}.txt", f"RUL_{s}.txt"]

# Public GitHub mirrors that host individual C-MAPSS files as raw text.
# Multiple mirrors tried in order — if one is down, the next is attempted.
GITHUB_MIRRORS = [
    # Mirror 1 — bagustris repo (commonly used in research)
    "https://raw.githubusercontent.com/bagustris/CMAPSS-Dataset/master/CMAPSSData/{filename}",
    # Mirror 2 — fallback
    "https://raw.githubusercontent.com/schwxd/CADA/master/data/CMAPSS/{filename}",
]


def files_already_present(dest: str) -> bool:
    return all(os.path.isfile(os.path.join(dest, f)) for f in EXPECTED_FILES)


def try_github_mirrors(dest: str) -> bool:
    """Try downloading each file from GitHub mirrors."""
    downloaded = []
    for filename in EXPECTED_FILES:
        success = False
        for mirror_template in GITHUB_MIRRORS:
            url = mirror_template.format(filename=filename)
            dest_path = os.path.join(dest, filename)
            try:
                print(f"    {filename} <- {url.split('/')[2]}...", end=" ", flush=True)
                urllib.request.urlretrieve(url, dest_path)
                # Quick sanity check: file should be > 1 KB and start with digits
                if os.path.getsize(dest_path) > 1000:
                    with open(dest_path, "r", errors="ignore") as f:
                        first_char = f.read(1)
                    if first_char.strip().isdigit():
                        print("OK")
                        success = True
                        downloaded.append(filename)
                        break
                    else:
                        print("bad content (got HTML?)")
                        os.remove(dest_path)
                else:
                    print("too small")
                    os.remove(dest_path)
            except Exception as e:
                print(f"failed ({e})")

        if not success:
            # Clean up partial files and abort
            for f in downloaded:
                try:
                    os.remove(os.path.join(dest, f))
                except Exception:
                    pass
            return False
    return True


def try_kaggle_api(dest: str) -> bool:
    """Try downloading via kaggle Python package."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("  Kaggle package not installed (pip install kaggle)")
        return False

    import subprocess
    zip_path = os.path.join(dest, "nasa-cmaps.zip")
    print("  Trying: kaggle datasets download behrad3d/nasa-cmaps ...")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "behrad3d/nasa-cmaps", "-p", dest],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Kaggle download failed: {result.stderr.strip()}")
        return False

    # Find and extract the downloaded zip
    for fname in os.listdir(dest):
        if fname.endswith(".zip"):
            zip_path = os.path.join(dest, fname)
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    for member in z.namelist():
                        base = os.path.basename(member)
                        if base in EXPECTED_FILES:
                            with z.open(member) as src:
                                with open(os.path.join(dest, base), "wb") as dst:
                                    dst.write(src.read())
                            print(f"    Extracted: {base}")
                os.remove(zip_path)
                return True
            except Exception as e:
                print(f"  Extraction failed: {e}")
    return False


def print_manual_instructions():
    print("""
  ============================================================
  MANUAL DOWNLOAD (2 minutes, no account needed if using
  the direct NASA link)
  ============================================================

  OPTION A — Kaggle (easiest, free account required):
    1. Go to: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
    2. Click the [Download] button (top right of the page)
    3. Save and unzip to: data\\cmapss\\

  OPTION B — NASA direct:
    1. Go to:
       https://phm.siam.org/wp-content/uploads/2019/07/CMAPSS.zip
    2. Extract all .txt files to: data\\cmapss\\

  After extraction, data\\cmapss\\ should contain these 12 files:
    train_FD001.txt   test_FD001.txt   RUL_FD001.txt
    train_FD002.txt   test_FD002.txt   RUL_FD002.txt
    train_FD003.txt   test_FD003.txt   RUL_FD003.txt
    train_FD004.txt   test_FD004.txt   RUL_FD004.txt

  Then run:
    python scripts/download_cmapss.py   (to verify)
    python scripts/train_cmapss.py --epochs 20
  ============================================================
""")


def verify(dest: str) -> None:
    print("\n  Verifying files...")
    ok = True
    for fname in EXPECTED_FILES:
        path = os.path.join(dest, fname)
        if os.path.isfile(path):
            size_kb = os.path.getsize(path) // 1024
            print(f"    OK  {fname}  ({size_kb} KB)")
        else:
            print(f"    MISSING  {fname}")
            ok = False
    if ok:
        print("\n  All 12 files present. Ready to train.")
        print("  Run: python scripts/train_cmapss.py --epochs 20")
    else:
        print("\n  Some files are missing. See instructions above.")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  C-MAPSS Dataset Setup")
    print("=" * 60)

    os.makedirs(DEST_DIR, exist_ok=True)

    if files_already_present(DEST_DIR):
        print(f"\n  All files already present in {DEST_DIR}/")
        verify(DEST_DIR)
        return

    # Method 1: GitHub mirrors
    print(f"\n  Method 1: GitHub mirrors (downloading {len(EXPECTED_FILES)} files)...")
    if try_github_mirrors(DEST_DIR) and files_already_present(DEST_DIR):
        print("\n  SUCCESS via GitHub mirror.")
        verify(DEST_DIR)
        return

    # Method 2: Kaggle API
    print("\n  Method 2: Kaggle API...")
    if try_kaggle_api(DEST_DIR) and files_already_present(DEST_DIR):
        print("\n  SUCCESS via Kaggle API.")
        verify(DEST_DIR)
        return

    # Method 3: Manual instructions
    print("\n  Automatic download did not succeed.")
    print_manual_instructions()

    # Check if user has manually downloaded while script was running
    if files_already_present(DEST_DIR):
        verify(DEST_DIR)


if __name__ == "__main__":
    main()
