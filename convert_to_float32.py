import numpy as np
from pathlib import Path

print("=" * 70)
print("Converting NPZ files from float64 to float32")
print("=" * 70)

original_dir = Path(r"C:\Develop\TensorFlow\data\raw_float64_original")
output_dir = Path(r"C:\Develop\TensorFlow\data\raw")

npz_files = sorted(original_dir.glob("*.npz"))

if not npz_files:
    print("[ERROR] No NPZ files found in raw_float64_original/")
    exit(1)

print(f"\nFound {len(npz_files)} NPZ files\n")

total_original = 0
total_compressed = 0

for npz_path in npz_files:
    filename = npz_path.name
    output_path = output_dir / filename

    try:
        print(f"Processing: {filename}...", end=" ", flush=True)

        # Load original file with float64
        original_data = np.load(str(npz_path), allow_pickle=True)
        original_size = npz_path.stat().st_size / 1e6
        total_original += original_size

        # Convert each array to float32
        converted_data = {}
        for key in original_data.files:
            arr = original_data[key]
            # Convert to float32 if it's numeric
            if np.issubdtype(arr.dtype, np.floating):
                converted_data[key] = arr.astype(np.float32)
            else:
                converted_data[key] = arr

        # Save as compressed NPZ with float32
        np.savez_compressed(str(output_path), **converted_data)

        output_size = output_path.stat().st_size / 1e6
        total_compressed += output_size

        reduction = ((original_size - output_size) / original_size) * 100
        print(f"OK  {original_size:.1f}MB -> {output_size:.1f}MB (-{reduction:.1f}%)")

    except Exception as e:
        print(f"ERROR: {str(e)}")

print("\n" + "=" * 70)
print("CONVERSION SUMMARY")
print("=" * 70)
print(f"Original total (float64):    {total_original:.1f} MB")
print(f"Converted total (float32):   {total_compressed:.1f} MB")
print(f"Space saved:                 {total_original - total_compressed:.1f} MB")
print(f"Reduction rate:              {((total_original - total_compressed) / total_original * 100):.1f}%")

print("\n[INFO] Original float64 files preserved in: data/raw_float64_original/")
print("[INFO] New float32 files created in: data/raw/")
print("=" * 70)
