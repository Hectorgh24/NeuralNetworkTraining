"""
Descarga y extrae los datasets pesados desde Google Drive (carpetas).

Requiere: pip install gdown

Uso:
    python scripts/download_data.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW32_DIR = DATA_DIR / "raw_float32"
RAW64_DIR = DATA_DIR / "raw_float64"

DRIVE_FOLDERS = {
    RAW32_DIR: "1cgHAgsyQ5hKyyDKike8JQ5FcMtOfsX7t",  # raw_float32
    RAW64_DIR: "13lBH4GVmGFVJ8cv81h7or4knRQW7wRMA",  # raw_float64
}


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[INFO] Instalando gdown ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def download_folder(target: Path, folder_id: str):
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        f"https://drive.google.com/drive/folders/{folder_id}",
        "--output",
        str(target),
        "--quiet",
    ]
    print(f"\n[INFO] Descargando en {target} ...")
    subprocess.check_call(cmd)


def main():
    ensure_gdown()
    for dest, fid in DRIVE_FOLDERS.items():
        download_folder(dest, fid)
    print("\n✓ Descarga completada. Datos listos en data/raw_float32 y data/raw_float64")


if __name__ == "__main__":
    main()
