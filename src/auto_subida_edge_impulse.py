"""
Script para automatizar la subida de los archivos CSV a Edge Impulse.
Soluciona el problema de tener que subir carpeta por carpeta manualmente para 
que Edge Impulse respete correctamente las 17 clases.

Estrategia:
  1. (Opcional) Borrar todos los datos existentes en el proyecto con --clean.
  2. Subir cada carpeta/clase de forma secuencial, con pausas entre clases
     y entre lotes para que Edge Impulse indexe correctamente todas las etiquetas.

Requisitos previos:
1. Instalar NodeJS y Edge Impulse CLI: `npm install -g edge-impulse-cli`
2. Puedes autenticarte usando este script pasando tu API KEY (ver ayuda con `-h`).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# API Keys de los proyectos en Edge Impulse (guardadas para no tener que pasarlas cada vez)
API_KEY_17 = "ei_52408dcfa5065d95bd5b0a2399fb1be66a0d635274ee7f63"
API_KEY_9 = "ei_e570664809a692461667a6f7910f5743f91b882bf7aa878e"



def parse_args():
    parser = argparse.ArgumentParser(description="Sube carpetas a Edge Impulse de manera automatizada.")
    parser.add_argument(
        "--dataset",
        choices=["entrenamiento_17_clases", "entrenamiento_9_clases"],
        default="entrenamiento_17_clases",
        help="Dataset a subir. Selecciona automáticamente la API Key y el directorio correctos."
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        required=False,
        default=None,
        help="API Key explícita (sobreescribe la seleccionada por --dataset)."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Borrar TODOS los datos existentes en el proyecto antes de subir."
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Segundos de espera entre la subida de cada clase (default: 5)."
    )
    parser.add_argument(
        "--chunk-delay",
        type=int,
        default=2,
        help="Segundos de espera entre cada lote de archivos dentro de una clase (default: 2)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Cantidad de archivos por lote (default: 20). Lotes más pequeños = menos errores."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simular la subida sin ejecutar comandos reales. Útil para verificar."
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=None,
        help="Directorio con las carpetas de clases exportadas. Por defecto depende del dataset."
    )
    return parser.parse_args()


def subir_clase(
    carpeta: Path,
    clase_nombre: str,
    api_key: str | None,
    chunk_size: int,
    chunk_delay: int,
    dry_run: bool,
    is_first_chunk: bool = False,
) -> bool:
    """Sube todos los CSVs de una carpeta como una sola clase a Edge Impulse."""
    archivos = sorted(carpeta.glob("*.csv"))
    if not archivos:
        print(f"   ⚠️  No hay archivos .csv en '{clase_nombre}', saltando.")
        return False

    total_archivos = len(archivos)
    total_chunks = (total_archivos + chunk_size - 1) // chunk_size
    print(f"   📁 {total_archivos} archivos → {total_chunks} lotes de máx. {chunk_size}")

    for idx in range(0, total_archivos, chunk_size):
        chunk = [f.absolute().as_posix() for f in archivos[idx:idx + chunk_size]]
        chunk_num = (idx // chunk_size) + 1

        comando = [
            "edge-impulse-uploader.cmd",
            "--category", "training",
            "--label", clase_nombre,
        ]
        if is_first_chunk and chunk_num == 1:
            comando.append("--clean")
            if not dry_run:
                print("   🗑️  Ejecutando limpieza del proyecto (--clean) en este lote...", end=" ", flush=True)

        if api_key:
            comando.extend(["--api-key", api_key])
        comando.extend(chunk)

        if dry_run:
            print(f"   [DRY-RUN] Lote {chunk_num}/{total_chunks}: {len(chunk)} archivos")
            continue

        try:
            print(f"   ⬆️  Lote {chunk_num}/{total_chunks} ({len(chunk)} archivos)...", end=" ", flush=True)
            subprocess.run(comando, check=True, capture_output=True, text=True, timeout=300)
            print("✅")
        except subprocess.CalledProcessError as exc:
            print("❌")
            print(f"   Error en lote {chunk_num} de '{clase_nombre}':")
            print(f"   {exc.stderr[:500] if exc.stderr else '(sin detalles)'}")
            return False
        except subprocess.TimeoutExpired:
            print("❌ (timeout)")
            print(f"   Tiempo de espera agotado en lote {chunk_num} de '{clase_nombre}'.")
            return False

        # Pausa entre lotes de la misma clase
        if chunk_num < total_chunks and chunk_delay > 0 and not dry_run:
            print(f"   ⏳ Esperando {chunk_delay}s entre lotes...", flush=True)
            time.sleep(chunk_delay)

    return True


def main():
    # Forzar UTF-8 en Windows para que los emojis se impriman correctamente
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    
    dataset_name = args.dataset
    api_key = args.api_key
    if not api_key:
        api_key = API_KEY_17 if dataset_name == "entrenamiento_17_clases" else API_KEY_9
        
    exports_dir = args.exports_dir
    if not exports_dir:
        exports_dir = Path(f"exports/edge_impulse_{dataset_name}.edgei")

    if not exports_dir.exists():
        print(f"[ERROR] No se encontró el directorio {exports_dir}.")
        sys.exit(1)

    carpetas = sorted([d for d in exports_dir.iterdir() if d.is_dir()])

    if not carpetas:
        print(f"[ERROR] No se encontraron carpetas con clases dentro de {exports_dir}.")
        sys.exit(1)

    print("=" * 60)
    print(f"🚀 SUBIDA AUTOMATIZADA A EDGE IMPULSE")
    print(f"   Clases encontradas: {len(carpetas)}")
    print(f"   Modo: {'DRY-RUN (simulación)' if args.dry_run else 'REAL'}")
    print(f"   Limpiar antes: {'Sí' if args.clean else 'No'}")
    print(f"   Tamaño de lote: {args.chunk_size}")
    print(f"   Delay entre clases: {args.delay}s")
    print(f"   Delay entre lotes: {args.chunk_delay}s")
    print("=" * 60)

    # Mostrar las clases que se van a subir
    print("\nClases a subir:")
    for i, c in enumerate(carpetas, 1):
        n_archivos = len(list(c.glob("*.csv")))
        print(f"   {i:2d}. {c.name} ({n_archivos} archivos)")

    # Paso 1: Limpiar datos existentes si se pidió
    if args.clean:
        print(f"\n🗑️  La limpieza (--clean) se ejecutará automáticamente junto con el primer lote de datos.")

    # Paso 2: Subir clase por clase con pausas
    resultados: dict[str, str] = {}
    total = len(carpetas)

    for i, carpeta in enumerate(carpetas, 1):
        clase_nombre = carpeta.name
        print(f"\n{'─' * 60}")
        print(f"[{i}/{total}] 📤 Subiendo clase: {clase_nombre}")
        print(f"{'─' * 60}")

        exito = subir_clase(
            carpeta=carpeta,
            clase_nombre=clase_nombre,
            api_key=api_key,
            chunk_size=args.chunk_size,
            chunk_delay=args.chunk_delay,
            dry_run=args.dry_run,
            is_first_chunk=(args.clean and i == 1),
        )

        resultados[clase_nombre] = "✅ OK" if exito else "❌ ERROR"

        # Pausa entre clases (excepto la última)
        if i < total and args.delay > 0 and not args.dry_run:
            print(f"\n⏳ Esperando {args.delay}s antes de la siguiente clase...")
            time.sleep(args.delay)

    # Paso 3: Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE SUBIDA")
    print("=" * 60)

    exitosas = 0
    fallidas = 0
    for clase, estado in resultados.items():
        print(f"   {clase:<25s} {estado}")
        if "OK" in estado:
            exitosas += 1
        else:
            fallidas += 1

    print(f"\n   Total: {exitosas} exitosas, {fallidas} fallidas de {total} clases.")

    if fallidas > 0:
        print("\n⚠️  Algunas clases fallaron. Revisa los errores arriba y reintenta.")
        sys.exit(1)
    else:
        print("\n🎉 ¡Todas las clases han sido subidas exitosamente a Edge Impulse!")
        print("   Verifica en Edge Impulse Studio que aparezcan las 17 etiquetas.")


if __name__ == "__main__":
    main()
