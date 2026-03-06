"""
Test rápido para verificar que el entorno funciona correctamente
"""
import numpy as np
import tensorflow as tf
from pathlib import Path

print("=" * 70)
print("TEST DE CONFIGURACIÓN")
print("=" * 70)

# 1. Verificar TensorFlow
print("\n[1/5] Verificando TensorFlow...")
print(f"✓ TensorFlow version: {tf.__version__}")

# 2. Verificar GPU (si está disponible)
print("\n[2/5] Verificando GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU detectada: {len(gpus)} GPU(s)")
    print(f"  {gpus[0]}")
else:
    print("⚠ GPU no detectada (usará CPU, será más lento)")

# 3. Cargar datos
print("\n[3/5] Cargando datos de ejemplo...")
try:
    data_dir = Path(r"C:\Develop\TensorFlow\data\raw")
    acc_data = np.load(data_dir / 'acc_data.npz', allow_pickle=True)['acc_data']
    acc_labels = np.load(data_dir / 'acc_labels.npz', allow_pickle=True)['acc_labels']
    print(f"✓ Datos cargados exitosamente")
    print(f"  - Muestras: {acc_data.shape[0]}")
    print(f"  - Características: {acc_data.shape[1]}")
    print(f"  - Tipo: {acc_data.dtype}")
except Exception as e:
    print(f"✗ Error cargando datos: {e}")
    exit(1)

# 4. Crear modelo simple
print("\n[4/5] Probando construcción de modelo...")
try:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(453,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(30, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"✓ Modelo creado exitosamente")
    print(f"  - Parámetros totales: {model.count_params():,}")
except Exception as e:
    print(f"✗ Error creando modelo: {e}")
    exit(1)

# 5. Entrenar 1 época con datos de ejemplo
print("\n[5/5] Realizando entrenamiento de prueba (1 época)...")
try:
    X_sample = acc_data[:100]
    y_sample = (acc_labels.flatten() - 1)[:100]  # Ajustar índices

    history = model.fit(
        X_sample, y_sample,
        epochs=1,
        batch_size=32,
        verbose=0
    )
    print(f"✓ Entrenamiento de prueba exitoso")
    print(f"  - Loss: {history.history['loss'][0]:.4f}")
    print(f"  - Accuracy: {history.history['accuracy'][0]:.4f}")
except Exception as e:
    print(f"✗ Error en entrenamiento de prueba: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✓ TODOS LOS TEST PASARON CORRECTAMENTE")
print("=" * 70)
print("\nPuedes ejecutar: python src/entrenamiento.py")
