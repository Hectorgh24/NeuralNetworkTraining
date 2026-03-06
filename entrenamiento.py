import scipy.io as sio
import numpy as np
import os

raw_dir = "/c/Develop/TensorFlow/data/raw"

# Leer archivos MAT y guardarlos como NPZ
for filename in os.listdir(raw_dir):
    if filename.endswith('_data.mat'):
        mat_file = os.path.join(raw_dir, filename)
        data = sio.loadmat(mat_file)
        
        # Guardar como NPZ (más compacto)
        base_name = filename.replace('.mat', '')
        npz_file = os.path.join(raw_dir, base_name + '.npz')
        np.savez_compressed(npz_file, **data)
        
        print(f"{filename} → {os.path.getsize(mat_file)/1e6:.1f}MB → {os.path.getsize(npz_file)/1e6:.1f}MB")