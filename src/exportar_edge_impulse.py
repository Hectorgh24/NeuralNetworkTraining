"""
Genera un ZIP y/o un directorio listo para subir a Edge Impulse con las 17 clases
del dataset UniMiB-SHAR (9 ADL + 8 caídas) en formato CSV por muestra.

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


# Las 17 clases de UniMiB-SHAR
# Se definen dinámicamente según el dataset
KEEP_IDS_17 = list(range(1, 18))
KEEP_IDS_9 = [3, 10, 11, 12, 13, 14, 15, 16, 17]


ID_TO_NAME_RAW = {
    1: "standing_up_fs",
    2: "standing_up_fl",
    3: "walk",
    4: "run",
    5: "going_up_stairs",
    6: "jump",
    7: "going_down_stairs",
    8: "lying_down_fs",
    9: "sitting_down",
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
    return re.sub(r"_{2,}", "_", slug).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta CSVs por muestra (17 clases) para Edge Impulse."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw_float32"),
        help="Directorio que contiene acc_data.npz y acc_labels.npz.",
    )
    parser.add_argument(
        "--dataset",
        choices=["entrenamiento_17_clases", "entrenamiento_9_clases"],
        default="entrenamiento_17_clases",
        help="Dataset a exportar: 17 clases completas o 9 clases (caminar + caídas).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta del zip generado. Por defecto: exports/edge_impulse_<dataset>.edgei.zip",
    )
    parser.add_argument(
        "--dir-output",
        type=Path,
        default=None,
        help="Ruta de la carpeta exportada. Por defecto, mismo nombre del zip sin .zip.",
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
        help="Frecuencia de muestreo original en Hz (por defecto 50 Hz).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Factor de diezmado. 2 guarda 1 de cada 2 muestras (50%% del tamaño).",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Cantidad de decimales al guardar accX/accY/accZ.",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=20.0,
        help="Valor máximo absoluto para limitar aceleraciones y evitar desbordes.",
    )
    parser.add_argument(
        "--compression",
        choices=["deflate", "lzma", "bz2"],
        default="deflate",
        help="Método de compresión del ZIP.",
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path(""),
        help="Prefijo opcional dentro del zip (ej. train/ o test/).",
    )
    return parser.parse_args()


def cargar_acc(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carga los arrays de datos y etiquetas desde el directorio especificado."""
    data_file = data_dir / "acc_data.npz"
    labels_file = data_dir / "acc_labels.npz"

    if not data_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            f"Faltan archivos en {data_dir}. Se requieren acc_data.npz y acc_labels.npz."
        )

    X = np.load(data_file, allow_pickle=True)["acc_data"]  # (N, length)
    y_raw = np.load(labels_file, allow_pickle=True)["acc_labels"]  # (N, 3)

    # Extraer columna de activity_id
    y = np.asarray(y_raw)[:, 0].astype(int)
    return X, y


def formatear_muestra_csv(
    serie_temporal: np.ndarray,
    dt_ms: int,
    decimals: int
) -> str:
    """Convierte una matriz Nx3 en un CSV string con la cabecera adecuada."""
    buf = io.StringIO()
    buf.write("timestamp,accX,accY,accZ\n")

    timestamps = np.arange(serie_temporal.shape[0]) * dt_ms
    fmt = f"{{:.{decimals}f}}"
    
    for t, (ax, ay, az) in zip(timestamps, serie_temporal):
        buf.write(f"{int(t)},{fmt.format(ax)},{fmt.format(ay)},{fmt.format(az)}\n")

    return buf.getvalue()


def procesar_serie_temporal(
    serie: np.ndarray,
    downsample: int,
    clip_val: float
) -> np.ndarray:
    """Aplica diezmado y clipping a la serie temporal."""
    serie = np.asarray(serie, dtype=np.float32).reshape(-1, 3)
    serie = serie[::downsample]
    if clip_val > 0:
        np.clip(serie, -clip_val, clip_val, out=serie)
    return serie


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
    keep_ids: list[int],
) -> tuple[dict[str, int], dict[int, int]]:
    """Genera la exportación de las series temporales en CSV/ZIP listos para Edge Impulse."""
    
    downsample = max(1, int(downsample))
    dt_ms = int(round(1000.0 / (sample_rate / downsample)))
    
    counts: dict[int, int] = defaultdict(int)
    skipped: dict[int, int] = defaultdict(int)

    save_zip = export_mode in ("zip", "both")
    save_dir = export_mode in ("dir", "both")

    comp_map = {
        "deflate": zipfile.ZIP_DEFLATED,
        "lzma": zipfile.ZIP_LZMA,
        "bz2": zipfile.ZIP_BZIP2,
    }
    ziptype = comp_map.get(compression, zipfile.ZIP_DEFLATED)

    dir_base: Path | None = None
    if save_dir:
        dir_base = output_dir or output_path.with_suffix("")
        dir_base.mkdir(parents=True, exist_ok=True)

    id_to_name = {k: slugify(v) for k, v in ID_TO_NAME_RAW.items()}
    manifest_lines = ["filename,label\n"]

    zf = zipfile.ZipFile(output_path, "w", compression=ziptype) if save_zip else None

    try:
        for row, label in zip(X, y):
            label = int(label)
            if label not in keep_ids:
                skipped[label] += 1
                continue

            # Procesar señal
            serie = procesar_serie_temporal(row, downsample, clip)
            csv_content = formatear_muestra_csv(serie, dt_ms, decimals)
            
            # Nombrar y mapear archivo
            class_name = id_to_name[label]
            counts[label] += 1
            filename = prefix / class_name / f"{class_name}_{counts[label]:05d}.csv"
            filename_str = filename.as_posix()

            # Guardar resultados
            if save_zip and zf is not None:
                zf.writestr(filename_str, csv_content)
                
            if save_dir and dir_base is not None:
                target_path = dir_base / filename
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(csv_content, encoding="utf-8")
                
            manifest_lines.append(f"{filename_str},{class_name}\n")

        # Escribir manifiesto
        manifest_content = "".join(manifest_lines)
        if save_zip and zf is not None:
            zf.writestr("labels_manifest.csv", manifest_content)
        if save_dir and dir_base is not None:
            (dir_base / "labels_manifest.csv").write_text(manifest_content, encoding="utf-8")
            
    finally:
        if zf is not None:
            zf.close()

    stats = {id_to_name[k]: v for k, v in dict(sorted(counts.items())).items()}
    return stats, dict(sorted(skipped.items()))


def main() -> None:
    args = parse_args()
    
    dataset_name = args.dataset
    if args.output is None:
        args.output = Path(f"exports/edge_impulse_{dataset_name}.edgei.zip")
    
    if args.dir_output is None:
        args.dir_output = args.output.with_suffix("")

    keep_ids = KEEP_IDS_17 if dataset_name == "entrenamiento_17_clases" else KEEP_IDS_9

    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
    try:
        X, y = cargar_acc(args.data_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except Exception as exc: 
        print(f"[ERROR] Error inesperado al leer: {exc}")
        sys.exit(1)

    print(f"Cargados {len(y)} registros de {args.data_dir}...")
    
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
        keep_ids=keep_ids,
    )

    total = sum(stats.values())
    print(f"\nFinalizado. Total de muestras exportadas: {total}")
    if args.export_mode in ("zip", "both"):
        print(f"ZIP generado: {args.output}")
    if args.export_mode in ("dir", "both"):
        dir_out = args.dir_output or args.output.with_suffix("")
        print(f"Directorio generado: {dir_out}")
        
    print("\nMuestras por clase:")
    for nombre, cantidad in stats.items():
        print(f" - {nombre:<20s}: {cantidad}")
        
    if skipped:
        skipped_total = sum(skipped.values())
        print(f"\n[ADVERTENCIA] Se saltaron {skipped_total} muestras.")
        
    print("\n¡Listo para subir a Edge Impulse!")


if __name__ == "__main__":
    main()
