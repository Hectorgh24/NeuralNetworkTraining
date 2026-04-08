"""
Genera un ZIP listo para subir a Edge Impulse con las 9 clases
del dataset UniMiB-SHAR (caminar + 8 caídas) en formato CSV por muestra.

Cada archivo CSV incluye encabezado: timestamp,accX,accY,accZ
con timestamps en milisegundos a 50 Hz (20 ms entre muestras).
Las clases se organizan en carpetas dentro del ZIP; Edge Impulse usa
el nombre de la carpeta como etiqueta.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


# IDs originales en UniMiB-SHAR que queremos conservar
KEEP_IDS = [3, 10, 11, 12, 13, 14, 15, 16, 17]

# Mapeo id original -> nombre de clase (folder en Edge Impulse)
ID_TO_NAME_RAW = {
    3: "walk",
    10: "fall_forward",
    11: "fall_backward",
    12: "fall_sideward_left",
    13: "fall_sideward_right",
    14: "fall_syncope",
    15: "fall_sitting",
    16: "fall_bending",
    17: "fall_hand",
}


def slugify(name: str) -> str:
    """Normaliza nombres de clase para que Edge Impulse los lea como una sola etiqueta."""
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_{2,}", "_", slug).strip("_")
    return slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta CSVs por muestra (9 clases) para Edge Impulse."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw_float32"),
        help="Directorio que contiene acc_data.npz y acc_labels.npz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/edge_impulse.edgei.zip"),
        help="Ruta del zip generado (si export-mode incluye zip).",
    )
    parser.add_argument(
        "--dir-output",
        type=Path,
        default=None,
        help="Ruta de la carpeta exportada (si export-mode incluye dir). Por defecto, mismo nombre del zip sin .zip.",
    )
    parser.add_argument(
        "--export-mode",
        choices=["zip", "dir", "both"],
        default="both",
        help="Generar solo zip, solo carpeta, o ambos. Recomendado: both.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=50.0,
        help="Frecuencia de muestreo en Hz (por defecto 50 Hz).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Factor de diezmado. 2 guarda 1 de cada 2 muestras (50% del tamaño).",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Cantidad de decimales al guardar accX/accY/accZ (reduce peso de texto).",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=20.0,
        help="Valor máximo absoluto para limitar aceleraciones y evitar desbordes en quantization.",
    )
    parser.add_argument(
        "--compression",
        choices=["deflate", "lzma", "bz2"],
        default="deflate",
        help="Método de compresión del ZIP. lzma comprime más pero tarda un poco más.",
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path(""),
        help="Prefijo opcional dentro del zip (ej. train/ o test/).",
    )
    return parser.parse_args()


def cargar_acc(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data_file = data_dir / "acc_data.npz"
    labels_file = data_dir / "acc_labels.npz"

    if not data_file.exists() or not labels_file.exists():
        raise FileNotFoundError("No se encontraron acc_data.npz o acc_labels.npz en el directorio indicado.")

    X = np.load(data_file, allow_pickle=True)["acc_data"]  # (N, 453)
    y_raw = np.load(labels_file, allow_pickle=True)["acc_labels"]  # (N, 3) -> [activity_id, subject_id, trial_id]

    # Solo necesitamos la columna de activity_id
    y = np.asarray(y_raw)[:, 0].astype(int)
    return X, y


def escribir_export(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    output_dir: Path | None,
    export_mode: str,
    sample_rate: float,
    downsample: int,
    decimals: int,
    compression: str,
    prefix: Path,
    clip: float,
) -> tuple[dict[str, int], dict[int, int]]:
    """
    Genera los CSVs en una carpeta, en un ZIP o en ambos.
    Edge Impulse puede rechazar el ZIP si se sube en modo "Select individual files",
    por eso ofrecemos exportar la carpeta directamente.
    """
    downsample = max(1, int(downsample))
    effective_rate = sample_rate / downsample
    dt_ms = int(round(1000.0 / effective_rate))
    counts: dict[int, int] = defaultdict(int)
    skipped: dict[int, int] = defaultdict(int)

    save_zip = export_mode in ("zip", "both")
    save_dir = export_mode in ("dir", "both")

    comp = zipfile.ZIP_DEFLATED
    if save_zip:
        compression_map = {
            "deflate": zipfile.ZIP_DEFLATED,
            "lzma": zipfile.ZIP_LZMA,
            "bz2": zipfile.ZIP_BZIP2,
        }
        comp = compression_map[compression]

    dir_base: Path | None = None
    if save_dir:
        dir_base = output_dir or output_path.with_suffix("")
        dir_base.mkdir(parents=True, exist_ok=True)

    # Normalizar nombres de clases y construir mapeo final
    id_to_name = {k: slugify(v) for k, v in ID_TO_NAME_RAW.items()}
    manifest_lines = ["filename,label\n"]

    zf_ctx = zipfile.ZipFile(output_path, "w", compression=comp) if save_zip else None
    if zf_ctx is None:
        class DummyZip:
            def writestr(self, *args, **kwargs):
                return None

            def close(self):
                return None

        zf_ctx = DummyZip()

    with zf_ctx as zf:
        for idx, (row, label) in enumerate(zip(X, y)):
            if label not in KEEP_IDS:
                skipped[int(label)] += 1
                continue

            serie = np.asarray(row, dtype=np.float32).reshape(-1, 3)
            # Reducir tamaño: diezmado simple
            serie = serie[::downsample]
            if clip > 0:
                np.clip(serie, -clip, clip, out=serie)
            timestamps = np.arange(serie.shape[0]) * dt_ms
            class_name = id_to_name[label]
            counts[label] += 1

            buf = io.StringIO()
            # Encabezado estricto numérico; Edge Impulse infiere la clase por carpeta.
            buf.write("timestamp,accX,accY,accZ\n")

            fmt = f"{{:.{decimals}f}}"
            for t, (ax, ay, az) in zip(timestamps, serie):
                ax_s = fmt.format(ax)
                ay_s = fmt.format(ay)
                az_s = fmt.format(az)
                buf.write(f"{int(t)},{ax_s},{ay_s},{az_s}\n")

            filename = prefix / class_name / f"{class_name}_{counts[label]:05d}.csv"
            # Asegurar separadores POSIX dentro del zip
            content = buf.getvalue()
            if save_zip:
                zf.writestr(filename.as_posix(), content)
            if save_dir and dir_base is not None:
                target_path = dir_base / filename
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content, encoding="utf-8")
            manifest_lines.append(f"{filename.as_posix()},{class_name}\n")

        # Manifesto opcional para verificación rápida de etiquetas en Edge Impulse
        manifest_content = "".join(manifest_lines)
        if save_zip:
            zf.writestr("labels_manifest.csv", manifest_content)
        if save_dir and dir_base is not None:
            (dir_base / "labels_manifest.csv").write_text(manifest_content, encoding="utf-8")

    stats = {id_to_name[k]: v for k, v in counts.items()}
    return stats, dict(sorted(skipped.items()))


def main() -> None:
    args = parse_args()
    try:
        X, y = cargar_acc(args.data_dir)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"[ERROR] {exc}")
        sys.exit(1)

    stats, skipped = escribir_export(
        X,
        y,
        output_path=args.output,
        output_dir=args.dir_output,
        export_mode=args.export_mode,
        sample_rate=args.sample_rate,
        downsample=args.downsample,
        decimals=args.decimals,
        compression=args.compression,
        prefix=args.prefix,
        clip=args.clip,
    )

    total = sum(stats.values())
    print(f"ZIP generado: {args.output} ({total} muestras)")
    for nombre, cantidad in stats.items():
        print(f" - {nombre:20s}: {cantidad}")
    if skipped:
        skipped_total = sum(skipped.values())
        print(f"Saltadas {skipped_total} muestras porque su activity_id no está en KEEP_IDS.")
        print(f"IDs permitidos: {sorted(KEEP_IDS)}")
        print(f"IDs saltados con conteo: {skipped}")
    print("Listo para subir a Edge Impulse (Upload existing data).")


if __name__ == "__main__":
    main()
