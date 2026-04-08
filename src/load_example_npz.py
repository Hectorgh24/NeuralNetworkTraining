"""
Ejemplo: Cómo cargar datos desde archivos NPZ para TensorFlow/Keras
"""

import numpy as np
from pathlib import Path

# Ruta a los datos
data_dir = Path(r"C:\Develop\TensorFlow\data\raw_float32")

# ===== OPCION 1: Cargar archivos NPZ (recomendado, más eficiente) =====
print("Cargando datos desde NPZ...")
acc_data = np.load(data_dir / 'acc_data.npz')
acc_labels = np.load(data_dir / 'acc_labels.npz')

print("\nContenido de acc_data.npz:")
print(acc_data.files)  # Muestra las claves disponibles

# Acceder a los datos
X = acc_data['__data__']  # Variable principal en el archivo
y = acc_labels['__data__']

print(f"\nForma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# ===== OPCION 2: Si prefieres los .mat originales =====
# import scipy.io as sio
# acc_data_mat = sio.loadmat(data_dir / 'acc_data.mat')
# X = acc_data_mat['acc_data']  # Clave principal
# y = acc_data_mat['__data__']  # O la que corresponda

# ===== CON TENSORFLOW/KERAS =====
"""
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Cargar datos
acc_data = np.load(data_dir / 'acc_data.npz')['__data__']
acc_labels = np.load(data_dir / 'acc_labels.npz')['__data__'].flatten()

# Dividir train/test
X_train, X_test, y_train, y_test = train_test_split(
    acc_data, acc_labels, test_size=0.2, random_state=42
)

# Crear modelo
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluar
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en test: {accuracy:.4f}")
"""
