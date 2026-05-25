import sys
import subprocess
import os
from pathlib import Path

def run_command(command, description):
    print(f"\n{'='*80}")
    print(f"▶ EJECUTANDO: {description}")
    print(f"▶ COMANDO: {' '.join(command)}")
    print(f"{'='*80}\n")
    
    try:
        subprocess.check_call(command)
        print(f"\n[✓] ÉXITO: {description}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[X] ERROR FATAL: Falló la ejecución de {description}.")
        print(f"Código de salida: {e.returncode}")
        print("Abortando la secuencia para evitar inconsistencias.")
        sys.exit(1)

def main():
    # Asegurar estar en el directorio raíz del proyecto
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)

    print("\n" + "#"*80)
    print(" INICIANDO PIPELINE AUTOMATIZADO COMPLETO (TENSORFLOW SHAR)")
    print("#"*80 + "\n")

    # 1. Actualizar e Instalar Dependencias
    run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Actualizando pip"
    )
    run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Instalando dependencias desde requirements.txt"
    )

    # 2. Descargar Datos Pre-Convertidos (Float32)
    run_command(
        [sys.executable, "scripts/download_data.py"],
        "Descargando datasets pre-convertidos desde Google Drive"
    )

    datasets = ["entrenamiento_17_clases", "entrenamiento_9_clases"]

    # 3. Ejecutar Pipeline para cada Dataset
    for ds in datasets:
        print(f"\n{'*'*80}")
        print(f"*** PROCESANDO DATASET: {ds.upper()} ***")
        print(f"{'*'*80}\n")

        # a. Entrenamiento
        run_command(
            [sys.executable, "src/entrenamiento.py", "--dataset", ds],
            f"Entrenamiento del modelo ({ds})"
        )

        # b. Generación de Reportes
        run_command(
            [sys.executable, "src/generar_reporte.py", "--dataset", ds],
            f"Generación de reportes PDF y gráficos HD ({ds})"
        )

        # c. Exportación a TFLite
        run_command(
            [sys.executable, "src/exportar_tflite.py", "--dataset", ds, "--output-dir", "exports/exportsTflite"],
            f"Exportación de modelo a TensorFlow Lite ({ds})"
        )

        # d. Exportación de Parámetros de Preprocesamiento
        run_command(
            [sys.executable, "exportar_parametros_preprocesamiento.py", "--dataset", ds],
            f"Exportación de parámetros StandardScaler a JSON ({ds})"
        )

    print("\n" + "#"*80)
    print(" ✓ PIPELINE COMPLETADO EXITOSAMENTE PARA TODOS LOS MODELOS ")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main()
