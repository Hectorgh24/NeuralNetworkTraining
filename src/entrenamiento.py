"""
Entrenamiento de Red Neuronal para Clasificación de Actividades Humanas
Dataset: UniMiB-SHAR (Accelerometer Activity Recognition)

Autor: Sistema de IA
Descripción: Red neuronal convolucional densa con regularización para clasificación
             multiclase de actividades usando datos de acelerómetro.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas
DATA_DIR = Path(r"C:\Develop\TensorFlow\data\raw")
MODELS_DIR = Path(r"C:\Develop\TensorFlow\models")
LOGS_DIR = Path(r"C:\Develop\TensorFlow\logs")

# Crear directorios si no existen
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Hiperparámetros
DATASET_NAME = 'acc'  # 'acc', 'adl', 'fall', 'two_classes'
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# ============================================================================
# CARGA DE DATOS
# ============================================================================

def cargar_datos(nombre_dataset):
    """
    Carga datos del dataset especificado.

    Args:
        nombre_dataset (str): 'acc', 'adl', 'fall' o 'two_classes'

    Returns:
        tuple: (X, y, nombres_clases)
    """
    print(f"\n[INFO] Cargando dataset: {nombre_dataset}")

    try:
        # Cargar datos
        data_file = DATA_DIR / f"{nombre_dataset}_data.npz"
        labels_file = DATA_DIR / f"{nombre_dataset}_labels.npz"
        names_file = DATA_DIR / f"{nombre_dataset}_names.npz"

        X = np.load(data_file, allow_pickle=True)[f"{nombre_dataset}_data"]
        y_raw = np.load(labels_file, allow_pickle=True)[f"{nombre_dataset}_labels"]
        nombres_clases = np.load(names_file, allow_pickle=True)[f"{nombre_dataset}_names"]

        print(f"✓ Datos cargados: {X.shape}")
        print(f"✓ Etiquetas cargadas: {y_raw.shape}")
        print(f"✓ Clases encontradas: {len(nombres_clases)}")

        return X, y_raw, nombres_clases

    except Exception as e:
        print(f"✗ Error cargando datos: {e}")
        raise


def preprocesar_datos(X, y):
    """
    Preprocesa los datos: normalización y conversión de etiquetas.

    Args:
        X (np.ndarray): Características (muestras, características)
        y (np.ndarray): Etiquetas (pueden ser one-hot o clase)

    Returns:
        tuple: (X_normalizado, y_clase)
    """
    print("\n[INFO] Preprocesando datos...")

    # Normalización usando StandardScaler
    scaler = StandardScaler()
    X_normalizado = scaler.fit_transform(X)
    X_normalizado = X_normalizado.astype(np.float32)

    # Convertir etiquetas a formato de clase (si es necesario)
    if len(y.shape) == 2:
        # Si son one-hot encoded, convertir a clase
        y_clase = np.argmax(y, axis=1)
    else:
        y_clase = y.flatten()

    # Ajustar índices de clases a partir de 0
    y_clase = y_clase - 1 if y_clase.min() > 0 else y_clase

    print(f"✓ Datos normalizados: media={X_normalizado.mean():.4f}, std={X_normalizado.std():.4f}")
    print(f"✓ Clases ajustadas: min={y_clase.min()}, max={y_clase.max()}")

    return X_normalizado, y_clase, scaler


def dividir_datos(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Divide datos en conjunto de entrenamiento, validación y prueba.

    Args:
        X: Características
        y: Etiquetas
        test_size: Proporción del conjunto de prueba
        val_size: Proporción del conjunto de validación (del entrenamiento)
        random_state: Semilla para reproducibilidad

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n[INFO] Dividiendo datos...")

    # Primera división: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Segunda división: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    print(f"✓ Entrenamiento: {X_train.shape[0]} muestras")
    print(f"✓ Validación:    {X_val.shape[0]} muestras")
    print(f"✓ Prueba:        {X_test.shape[0]} muestras")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# CONSTRUCCIÓN DEL MODELO
# ============================================================================

def construir_modelo(input_shape, num_clases):
    """
    Construye la arquitectura de la red neuronal profunda con regularización.

    Arquitectura:
    - Capas densas progresivas con regularización L2
    - Activación ReLU en capas ocultas, Softmax en salida
    - Dropout para evitar overfitting
    - Batch Normalization para estabilidad

    Args:
        input_shape (int): Número de características de entrada
        num_clases (int): Número de clases de salida

    Returns:
        keras.Model: Modelo compilado
    """
    print("\n[INFO] Construyendo modelo...")

    model = models.Sequential([
        # Capa de entrada
        layers.Input(shape=(input_shape,)),

        # Bloque 1
        layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='dense_1'
        ),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.3, name='dropout_1'),

        # Bloque 2
        layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='dense_2'
        ),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.3, name='dropout_2'),

        # Bloque 3
        layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='dense_3'
        ),
        layers.BatchNormalization(name='bn_3'),
        layers.Dropout(0.2, name='dropout_3'),

        # Bloque 4
        layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='dense_4'
        ),
        layers.Dropout(0.2, name='dropout_4'),

        # Capa de salida
        layers.Dense(
            num_clases,
            activation='softmax',
            name='output'
        )
    ])

    # Compilación del modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    print("✓ Modelo construido exitosamente")
    print("\nArquitectura del modelo:")
    model.summary()

    return model


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def entrenar_modelo(model, X_train, y_train, X_val, y_val, epochs=100):
    """
    Entrena el modelo con callbacks para mejorar el rendimiento.

    Callbacks utilizados:
    - EarlyStopping: Detiene el entrenamiento si no hay mejoría
    - ReduceLROnPlateau: Reduce learning rate si se estanca
    - ModelCheckpoint: Guarda el mejor modelo
    - TensorBoard: Monitoreo en tiempo real

    Args:
        model: Modelo de Keras
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        epochs: Número máximo de épocas

    Returns:
        keras.callbacks.History: Historial de entrenamiento
    """
    print("\n[INFO] Iniciando entrenamiento...")

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=MODELS_DIR / f"{DATASET_NAME}_mejor_modelo.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    tensorboard = callbacks.TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=1,
        write_graph=True
    )

    # Entrenamiento
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr, model_checkpoint, tensorboard],
        verbose=1
    )

    print("\n✓ Entrenamiento completado")
    return history


# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluar_modelo(model, X_test, y_test, nombres_clases):
    """
    Evalúa el modelo en el conjunto de prueba y genera métricas.

    Métricas calculadas:
    - Precisión general (Accuracy)
    - Precisión, Recall y F1-Score por clase
    - Matriz de confusión

    Args:
        model: Modelo entrenado
        X_test, y_test: Datos de prueba
        nombres_clases: Nombres de las clases

    Returns:
        dict: Métricas de evaluación
    """
    print("\n[INFO] Evaluando modelo en conjunto de prueba...")

    # Predicciones
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Métricas generales
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n{'='*50}")
    print(f"MÉTRICAS DE EVALUACIÓN")
    print(f"{'='*50}")
    print(f"Precisión (Accuracy): {accuracy:.4f}")
    print(f"Precisión (Weighted): {precision:.4f}")
    print(f"Recall (Weighted):    {recall:.4f}")
    print(f"F1-Score (Weighted):  {f1:.4f}")

    # Reporte por clase
    print(f"\n{'='*50}")
    print(f"REPORTE POR CLASE")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=nombres_clases, zero_division=0))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Guardar métricas
    metricas = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'num_muestras_test': int(len(y_test)),
        'clases': [str(c) for c in nombres_clases]
    }

    return metricas, y_pred, y_test, cm


def guardar_metricas(metricas, historico, modelo_nombre):
    """
    Guarda las métricas en un archivo JSON.

    Args:
        metricas (dict): Diccionario con métricas
        historico: Historial de entrenamiento
        modelo_nombre (str): Nombre del modelo
    """
    metricas['historico'] = {
        'loss': [float(x) for x in historico.history['loss']],
        'accuracy': [float(x) for x in historico.history['accuracy']],
        'val_loss': [float(x) for x in historico.history['val_loss']],
        'val_accuracy': [float(x) for x in historico.history['val_accuracy']]
    }

    ruta_metricas = MODELS_DIR / f"{modelo_nombre}_metricas.json"
    with open(ruta_metricas, 'w') as f:
        json.dump(metricas, f, indent=4)

    print(f"\n✓ Métricas guardadas en: {ruta_metricas}")


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def graficar_historico(history, dataset_name):
    """
    Grafica el historial de entrenamiento (loss y accuracy).

    Args:
        history: Objeto History de Keras
        dataset_name: Nombre del dataset para el título
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de pérdida (Loss)
    axes[0].plot(history.history['loss'], label='Pérdida de Entrenamiento', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Pérdida de Validación', linewidth=2)
    axes[0].set_xlabel('Épocas', fontsize=12)
    axes[0].set_ylabel('Pérdida', fontsize=12)
    axes[0].set_title(f'Pérdida - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Gráfico de precisión (Accuracy)
    axes[1].plot(history.history['accuracy'], label='Precisión de Entrenamiento', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Precisión de Validación', linewidth=2)
    axes[1].set_xlabel('Épocas', fontsize=12)
    axes[1].set_ylabel('Precisión', fontsize=12)
    axes[1].set_title(f'Precisión - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    ruta_grafico = LOGS_DIR / f"{dataset_name}_historico.png"
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Gráfico de histórico guardado: {ruta_grafico}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Función principal que ejecuta el pipeline completo de entrenamiento.
    """
    print("\n" + "="*70)
    print("ENTRENAMIENTO DE RED NEURONAL - CLASIFICACIÓN DE ACTIVIDADES")
    print("="*70)

    try:
        # 1. Cargar datos
        X, y, nombres_clases = cargar_datos(DATASET_NAME)

        # 2. Preprocesar
        X_proc, y_proc, scaler = preprocesar_datos(X, y)

        # 3. Dividir datos
        X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(
            X_proc, y_proc, TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE
        )

        # 4. Construir modelo
        num_clases = len(np.unique(y_proc))
        input_shape = X_proc.shape[1]
        model = construir_modelo(input_shape, num_clases)

        # 5. Entrenar
        history = entrenar_modelo(model, X_train, y_train, X_val, y_val, EPOCHS)

        # 6. Evaluar
        metricas, y_pred, y_test_eval, cm = evaluar_modelo(
            model, X_test, y_test, nombres_clases
        )

        # 7. Guardar modelo y métricas
        model.save(MODELS_DIR / f"{DATASET_NAME}_modelo.h5")
        print(f"\n✓ Modelo guardado: {MODELS_DIR}/{DATASET_NAME}_modelo.h5")

        guardar_metricas(metricas, history, DATASET_NAME)

        # 8. Visualizar
        graficar_historico(history, DATASET_NAME)

        # 9. Guardar artefactos para el reporte
        np.save(LOGS_DIR / f"{DATASET_NAME}_y_test.npy", y_test_eval)
        np.save(LOGS_DIR / f"{DATASET_NAME}_y_pred.npy", y_pred)
        np.save(LOGS_DIR / f"{DATASET_NAME}_matriz_confusion.npy", cm)

        print("\n" + "="*70)
        print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)

        return model, history, metricas, cm

    except Exception as e:
        print(f"\n✗ Error en entrenamiento: {e}")
        raise


if __name__ == "__main__":
    main()
