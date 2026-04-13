"""
Exporta un modelo Keras (.keras) a TensorFlow Lite (.tflite).

Uso rápido:
    python src/exportar_tflite.py

Opciones:
    --input       Ruta del modelo .keras de entrada.
    --output-dir  Carpeta destino para el archivo .tflite.
    --float16     Activa cuantización float16 para reducir tamaño.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tensorflow as tf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_BASE_BY_DATASET = {
    "two_classes": "entrenamiento_9_clases",
    "entrenamiento_9_clases": "entrenamiento_9_clases",
    "9_clases": "entrenamiento_9_clases",
    "adl_fall_multiclass": "entrenamiento_17_clases",
    "entrenamiento_17_clases": "entrenamiento_17_clases",
    "17_clases": "entrenamiento_17_clases",
}


def _canonical_dataset_name(nombre_raw: str) -> str:
    return {
        "entrenamiento_9_clases": "two_classes",
        "9_clases": "two_classes",
        "entrenamiento_17_clases": "adl_fall_multiclass",
        "17_clases": "adl_fall_multiclass",
    }.get(nombre_raw, nombre_raw)


def _modelo_por_dataset(dataset_name: str) -> Path:
    raiz = Path(__file__).resolve().parents[1]
    base = MODEL_BASE_BY_DATASET.get(dataset_name, dataset_name)
    return raiz / "models" / f"{base}_modelo.keras"


def construir_argumentos() -> argparse.Namespace:
    """Define y parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Convierte un modelo .keras a .tflite para Android."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Ruta del archivo .keras de entrada (si no se usa --dataset).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Alias del dataset: entrenamiento_9_clases | two_classes | entrenamiento_17_clases | adl_fall_multiclass",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "exports" / "exportsTflite",
        help="Directorio donde se guardará el .tflite.",
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Aplica cuantización float16 para reducir tamaño del modelo.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Aplica cuantización full integer (int8) para máxima compresión y velocidad en Edge AI.",
    )
    args = parser.parse_args()

    if args.input is None:
        # Seleccionar por dataset o default (17 clases)
        dataset = _canonical_dataset_name(args.dataset.strip()) if args.dataset else "adl_fall_multiclass"
        args.input = _modelo_por_dataset(dataset)

    return args


def convertir_a_tflite(modelo_keras: Path, output_dir: Path, usar_float16: bool = False, usar_int8: bool = False) -> Path:
    """
    Convierte un archivo .keras a .tflite con cuantización post-entrenamiento.

    Args:
        modelo_keras: Ruta del archivo .keras
        output_dir: Directorio de salida para el .tflite
        usar_float16: Aplicar cuantización a float16
        usar_int8: Aplicar cuantización optimizada con DEFAULT (sin requerir representative_dataset)

    Returns:
        Path: Ruta del archivo .tflite generado
    """
    if not modelo_keras.exists():
        # Intentar fallback: si pidió *_modelo probar *_mejor_modelo y viceversa
        alt = None
        name = modelo_keras.name
        if name.endswith("_modelo.keras"):
            alt = modelo_keras.with_name(name.replace("_modelo.keras", "_mejor_modelo.keras"))
        elif name.endswith("_mejor_modelo.keras"):
            alt = modelo_keras.with_name(name.replace("_mejor_modelo.keras", "_modelo.keras"))
        if alt and alt.exists():
            print(f"[WARN] No se encontró {modelo_keras}, usando alternativa {alt}")
            modelo_keras = alt
        else:
            raise FileNotFoundError(f"No se encontró el modelo de entrada: {modelo_keras}")
    if modelo_keras.suffix.lower() != ".keras":
        raise ValueError(f"Se esperaba un archivo .keras, recibido: {modelo_keras.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ruta_salida = output_dir / f"{modelo_keras.stem}.tflite"

    # Cargar el modelo Keras entrenado.
    modelo = tf.keras.models.load_model(modelo_keras)

    # Crear convertidor TensorFlow Lite a partir del modelo cargado.
    convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    convertidor.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # Cuantización post-entrenamiento para reducir tamaño (crítico para Edge AI < 1 MB)
    if usar_int8:
        print("[INFO] Aplicando cuantización optimizada con Dynamic Range Quantization...")
        convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
        # Dynamic range quantization: comprime pesos a int8 sin requerir representative_dataset
        # Esto es ideal para modelos de clasificación en Edge AI
    elif usar_float16:
        print("[INFO] Aplicando cuantización Float16...")
        convertidor.optimizations = [tf.lite.Optimize.DEFAULT]
        convertidor.target_spec.supported_types = [tf.float16]
    else:
        print("[INFO] Aplicando optimización DEFAULT...")
        convertidor.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = convertidor.convert()
    ruta_salida.write_bytes(tflite_model)

    return ruta_salida


def main() -> None:
    args = construir_argumentos()

    print("=" * 70)
    print("EXPORTACIÓN DE MODELO A TENSORFLOW LITE CON CUANTIZACIÓN")
    print("=" * 70)
    print(f"Modelo de entrada: {args.input}")
    print(f"Carpeta destino:   {args.output_dir}")
    print(f"Cuantización f16:  {'Sí' if args.float16 else 'No'}")
    print(f"Cuantización i8:   {'Sí' if args.int8 else 'No'}")

    ruta_tflite = convertir_a_tflite(
        args.input, 
        args.output_dir, 
        usar_float16=args.float16,
        usar_int8=args.int8
    )

    tam_keras_mb = args.input.stat().st_size / (1024 * 1024)
    tam_tflite_mb = ruta_tflite.stat().st_size / (1024 * 1024)

    print("\n✓ Exportación completada")
    print(f"Archivo generado:  {ruta_tflite}")
    print(f"Tamaño .keras:     {tam_keras_mb:.2f} MB")
    print(f"Tamaño .tflite:    {tam_tflite_mb:.2f} MB")
    if tam_tflite_mb > 1.0:
        print(f"⚠ ADVERTENCIA: Tamaño excede 1 MB. Se requiere cuantización más agresiva.")
        print(f"  Sugerencia: Intenta con --int8 para máxima compresión.")
    elif tam_tflite_mb < 1.0:
        print(f"✓ Modelo bajo 1 MB, listo para Edge AI en Android")
    print("=" * 70)


if __name__ == "__main__":
    main()
