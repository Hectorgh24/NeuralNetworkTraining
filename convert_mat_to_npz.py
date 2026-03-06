import scipy.io as sio
import numpy as np
import sys
from pathlib import Path

# Usar Path para compatibilidad con Windows
raw_dir = Path(r"C:\Develop\TensorFlow\data\raw")

print("=" * 70)
print("Convirtiendo archivos MAT a NPZ (sin borrar originales)")
print("=" * 70)

# Crear lista de archivos a convertir
mat_files = [f for f in raw_dir.glob('*.mat')]

if not mat_files:
    print("[ERROR] No hay archivos .mat en data/raw/")
    sys.exit(1)

print(f"\nEncontrados {len(mat_files)} archivos .mat\n")

total_original = 0
total_compressed = 0
errors = []

# Convertir cada archivo
for mat_path in sorted(mat_files):
    filename = mat_path.name
    npz_path = mat_path.parent / filename.replace('.mat', '.npz')

    try:
        # Obtener tamaño original
        original_size = mat_path.stat().st_size / 1e6
        total_original += original_size

        # Leer archivo MAT
        print(f"Leyendo: {filename}...", end=" ", flush=True)
        mat_data = sio.loadmat(str(mat_path))

        # Guardar como NPZ comprimido
        print(f"Comprimiendo...", end=" ", flush=True)
        np.savez_compressed(str(npz_path), **mat_data)

        # Obtener tamaño comprimido
        compressed_size = npz_path.stat().st_size / 1e6
        total_compressed += compressed_size

        # Calcular porcentaje de reducción
        reduction = ((original_size - compressed_size) / original_size) * 100

        print(f"OK  {original_size:.1f}MB -> {compressed_size:.1f}MB (-{reduction:.1f}%)")

    except Exception as e:
        error_msg = f"ERROR {filename}: {str(e)}"
        print(error_msg)
        errors.append(error_msg)

print("\n" + "=" * 70)
print("RESUMEN DE CONVERSION")
print("=" * 70)
print(f"Archivos originales (MAT):    {total_original:.1f} MB")
print(f"Archivos comprimidos (NPZ):   {total_compressed:.1f} MB")
print(f"Espacio ahorrado:             {total_original - total_compressed:.1f} MB")
print(f"Reduccion total:              {((total_original - total_compressed) / total_original * 100):.1f}%")

if errors:
    print("\n[ATENCIÓN] ERRORES DURANTE LA CONVERSION:")
    for error in errors:
        print(f"  {error}")
else:
    print("\n[EXITO] Conversion completada correctamente!")
    print("\n[INFO] Los archivos .mat originales se mantienen intactos")
    print("[INFO] Nuevos archivos .npz creados junto a los MAT")

print("=" * 70)
