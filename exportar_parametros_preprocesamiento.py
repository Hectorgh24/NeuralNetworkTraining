from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from entrenamiento import cargar_datos, preprocesar_datos  # type: ignore

CANONICAL_DATASETS: Dict[str, str] = {
    "entrenamiento_9_clases": "two_classes",
    "entrenamiento_17_clases": "adl_fall_multiclass",
}

OUTPUT_FILENAMES: Dict[str, str] = {
    "two_classes": "scaler_9_clases.json",
    "adl_fall_multiclass": "scaler_17_clases.json",
}


def parse_arguments() -> argparse.Namespace:
    """Parsea argumentos CLI para la exportación de parámetros de preprocesamiento."""
    parser = argparse.ArgumentParser(
        description=(
            "Exporta parámetros de preprocesamiento del scaler para el dataset "
            "seleccionado y genera opcionalmente un archivo Kotlin." 
        )
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        required=True,
        help=(
            "Nombre del dataset a procesar. Valores válidos: "
            "entrenamiento_9_clases, entrenamiento_17_clases."
        ),
    )
    parser.add_argument(
        "--generate-kotlin",
        dest="generate_kotlin",
        action="store_true",
        help="Genera el archivo DataPreprocessor.kt además del JSON.",
    )
    return parser.parse_args()


def canonicalize_dataset(dataset_name: str) -> str:
    """Normaliza aliases de dataset a nombres canónicos esperados."""
    lower_name = dataset_name.strip().lower()
    return CANONICAL_DATASETS.get(lower_name, lower_name)


def build_export_paths(dataset_name: str) -> Path:
    """Construye la ruta de salida del archivo JSON en el directorio exports/parametros-preprocesamiento."""
    filename = OUTPUT_FILENAMES.get(dataset_name)
    if filename is None:
        raise ValueError(
            f"Dataset no soportado para exportación: {dataset_name}. "
            f"Use one of: {', '.join(sorted(set(OUTPUT_FILENAMES.keys())))}"
        )
    exports_dir = ROOT_DIR / "exports" / "parametros-preprocesamiento"
    exports_dir.mkdir(parents=True, exist_ok=True)
    return exports_dir / filename


def build_kotlin_source(mean_values: List[float], scale_values: List[float]) -> str:
    """Genera el contenido Kotlin para los vectores de media y escala."""
    def format_float_list(values: List[float]) -> str:
        lines: List[str] = []
        for start in range(0, len(values), 10):
            chunk = values[start : start + 10]
            line = ", ".join(f"{value:.8f}f" for value in chunk)
            lines.append(f"    {line}")
        return ",\n".join(lines)

    mean_block = format_float_list(mean_values)
    scale_block = format_float_list(scale_values)

    return (
        "object DataPreprocessor {\n"
        "    val means = floatArrayOf(\n"
        f"{mean_block}\n"
        "    )\n\n"
        "    val stds = floatArrayOf(\n"
        f"{scale_block}\n"
        "    )\n"
        "}\n"
    )


def export_json(json_path: Path, mean_values: List[float], scale_values: List[float], dataset_name: str) -> None:
    """Escribe los parámetros del scaler a un archivo JSON."""
    if json_path.exists():
        print(f"✓ El archivo JSON ya existe y no será sobreescrito: {json_path}")
        return

    payload = {
        "dataset": dataset_name,
        "scaler": {
            "mean": mean_values,
            "scale": scale_values,
        },
    }
    try:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"✓ Archivo JSON generado: {json_path}")
    except OSError as exc:
        raise IOError(f"No se pudo escribir el archivo JSON '{json_path}': {exc}") from exc


def export_kotlin(kotlin_path: Path, mean_values: List[float], scale_values: List[float]) -> None:
    """Escribe el archivo DataPreprocessor.kt con los parámetros del scaler."""
    content = build_kotlin_source(mean_values, scale_values)
    try:
        with kotlin_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        print(f"✓ Archivo Kotlin generado: {kotlin_path}")
    except OSError as exc:
        raise IOError(f"No se pudo escribir el archivo Kotlin '{kotlin_path}': {exc}") from exc


def main() -> int:
    """Punto de entrada principal del script."""
    args = parse_arguments()
    dataset_name = canonicalize_dataset(args.dataset)

    if dataset_name not in OUTPUT_FILENAMES:
        print(
            f"ERROR: Dataset inválido '{args.dataset}'. Use valores como 'entrenamiento_9_clases' o 'entrenamiento_17_clases'."
        )
        return 1

    json_path = build_export_paths(dataset_name)
    kotlin_path = ROOT_DIR / "DataPreprocessor.kt"

    try:
        X, y, _ = cargar_datos(dataset_name)
        _, _, scaler = preprocesar_datos(X, y)

        mean_values = [float(value) for value in scaler.mean_.tolist()]
        scale_values = [float(value) for value in scaler.scale_.tolist()]

        export_json(json_path, mean_values, scale_values, dataset_name)

        if args.generate_kotlin:
            export_kotlin(kotlin_path, mean_values, scale_values)

    except FileNotFoundError as exc:
        print(f"ERROR: Archivo no encontrado: {exc}")
        return 2
    except ValueError as exc:
        print(f"ERROR de valor: {exc}")
        return 3
    except IOError as exc:
        print(f"ERROR de E/S: {exc}")
        return 4
    except Exception as exc:
        print(f"ERROR inesperado: {exc}")
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
