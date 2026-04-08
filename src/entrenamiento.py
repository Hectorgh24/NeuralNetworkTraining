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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
import sys
import io
import numbers

# Evita fallos de impresión en consolas Windows con codificación cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas
DATA_DIR = Path(r"C:\Develop\TensorFlow\data\raw_float32")
MODELS_DIR = Path(r"C:\Develop\TensorFlow\models")
LOGS_DIR = Path(r"C:\Develop\TensorFlow\logs")

# Crear directorios si no existen
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Hiperparámetros
DATASET_NAME_RAW = os.getenv(
    "DATASET_NAME",
    "adl_fall_multiclass"
).strip()
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# IDs originales (dataset ACC) que se deben conservar para el nuevo entrenamiento
FALL_IDS_ORIGINALES = [10, 11, 12, 13, 14, 15, 16, 17]
WALKING_ID_ORIGINAL = 3
FALL_WALK_IDS = [WALKING_ID_ORIGINAL] + FALL_IDS_ORIGINALES
MODEL_BASE_BY_DATASET = {
    "two_classes": "entrenamiento_9_clases",
    "entrenamiento_9_clases": "entrenamiento_9_clases",
    "9_clases": "entrenamiento_9_clases",
    "adl_fall_multiclass": "entrenamiento_17_clases",
    "entrenamiento_17_clases": "entrenamiento_17_clases",
    "17_clases": "entrenamiento_17_clases",
    "acc": "entrenamiento_17_clases",
}

# ============================================================================
# CARGA DE DATOS
# ============================================================================

def _normalizar_nombres_clases(nombres_clases):
    """
    Convierte nombres de clase a una lista plana de strings.
    """
    nombres_clases = np.asarray(nombres_clases, dtype=object).reshape(-1)
    return [
        str(c.item()) if isinstance(c, np.ndarray) and c.size == 1 else str(c)
        for c in nombres_clases
    ]


def _cargar_nombres_acc_originales():
    """
    Lee los nombres originales (17 clases) desde acc_names.npz.
    """
    try:
        names_file = DATA_DIR / "acc_names.npz"
        nombres_raw = np.load(names_file, allow_pickle=True)["acc_names"]
        nombres = _normalizar_nombres_clases(nombres_raw)
        # El archivo trae dos filas (nombre largo y corto); nos quedamos con los 17 primeros.
        if len(nombres) >= 17:
            nombres = nombres[:17]
        return nombres
    except Exception as exc:
        print(f"⚠ No se pudieron leer nombres originales de ACC: {exc}")
        return []


ACC_NOMBRES_ORIGINALES = _cargar_nombres_acc_originales()


def _nombre_por_id_acc(idx):
    """
    Devuelve el nombre original asociado a un id (1..17) del dataset ACC.
    """
    if 1 <= idx <= len(ACC_NOMBRES_ORIGINALES):
        return ACC_NOMBRES_ORIGINALES[idx - 1]
    return f"clase_{idx}"


# Mapeo dinámico id original -> índice nuevo 0..N
ACC_ID_TO_NEW = {orig_id: idx for idx, orig_id in enumerate(FALL_WALK_IDS)}
TWO_CLASSES_NOMBRES = [_nombre_por_id_acc(i) for i in FALL_WALK_IDS]
TWO_CLASSES_DETALLE = [
    {
        "clase": _nombre_por_id_acc(orig_id),
        "ids_originales": [orig_id],
        "etiquetas_originales": [_nombre_por_id_acc(orig_id)]
    }
    for orig_id in FALL_WALK_IDS
]


def _nombre_base_modelo(dataset_name):
    """
    Devuelve el prefijo de nombre a usar para archivos de modelo según dataset.
    """
    return MODEL_BASE_BY_DATASET.get(dataset_name, dataset_name)


def _parse_args_cli():
    """
    Permite especificar dataset por argumento CLI: --dataset nombre
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", "--data", dest="dataset", default=None)
    args, _ = parser.parse_known_args()
    return args.dataset


def _limpiar_modelos_antiguos(dataset_name):
    """
    Elimina archivos .keras legacy que usaban el nombre del dataset original.
    Mantiene los archivos con el nuevo esquema (entrenamiento_9/17_clases).
    """
    legado = [
        MODELS_DIR / f"{dataset_name}_modelo.keras",
        MODELS_DIR / f"{dataset_name}_mejor_modelo.keras",
    ]
    for ruta in legado:
        try:
            ruta.unlink(missing_ok=True)
        except Exception:
            pass


def _canonical_dataset_name(nombre_raw):
    """
    Normaliza nombres de dataset aceptando alias más legibles.
    """
    aliases = {
        "entrenamiento_9_clases": "two_classes",
        "9_clases": "two_classes",
        "entrenamiento9clases": "two_classes",
        "entrenamiento_17_clases": "adl_fall_multiclass",
        "17_clases": "adl_fall_multiclass",
        "entrenamiento17clases": "adl_fall_multiclass",
    }
    return aliases.get(nombre_raw, nombre_raw)


def _extraer_etiqueta_clase(y_raw):
    """
    Extrae la etiqueta de clase desde diferentes formatos.

    Para UniMiB-SHAR (N,3), la clase está en la primera columna.
    """
    y_raw = np.asarray(y_raw)

    if y_raw.ndim == 1:
        return y_raw.astype(np.int32)

    if y_raw.ndim == 2:
        # One-hot clásico
        if y_raw.shape[1] > 1:
            es_binario = np.isin(y_raw, [0, 1]).all()
            suma_uno = np.all(y_raw.sum(axis=1) == 1)
            if es_binario and suma_uno:
                return np.argmax(y_raw, axis=1).astype(np.int32)

        # Formato UniMiB-SHAR: [activity_id, subject_id, trial_id]
        return y_raw[:, 0].astype(np.int32)

    raise ValueError(f"Formato de etiquetas no soportado: {y_raw.shape}")


def _cargar_dataset_base(nombre_dataset):
    """
    Carga un dataset base desde archivos NPZ de UniMiB-SHAR.
    """
    data_file = DATA_DIR / f"{nombre_dataset}_data.npz"
    labels_file = DATA_DIR / f"{nombre_dataset}_labels.npz"
    names_file = DATA_DIR / f"{nombre_dataset}_names.npz"

    X = np.load(data_file, allow_pickle=True)[f"{nombre_dataset}_data"]
    y_raw = np.load(labels_file, allow_pickle=True)[f"{nombre_dataset}_labels"]
    nombres_clases = np.load(names_file, allow_pickle=True)[f"{nombre_dataset}_names"]
    nombres_clases = _normalizar_nombres_clases(nombres_clases)

    return X, y_raw, nombres_clases


def cargar_datos(nombre_dataset):
    """
    Carga datos del dataset especificado.

    Args:
        nombre_dataset (str): 'adl_fall_multiclass', 'acc', 'adl', 'fall' o 'two_classes'

    Returns:
        tuple: (X, y, nombres_clases)
    """
    print(f"\n[INFO] Cargando dataset: {nombre_dataset}")

    try:
        if nombre_dataset == 'adl_fall_multiclass':
            X_adl, y_adl_raw, nombres_adl = _cargar_dataset_base('adl')
            X_fall, y_fall_raw, nombres_fall = _cargar_dataset_base('fall')

            y_adl = _extraer_etiqueta_clase(y_adl_raw)
            y_fall = _extraer_etiqueta_clase(y_fall_raw)

            # Ajustar a índices consecutivos para clasificación multiclase conjunta.
            y_adl = y_adl - y_adl.min()
            y_fall = y_fall - y_fall.min() + len(nombres_adl)

            X = np.concatenate([X_adl, X_fall], axis=0)
            y_raw = np.concatenate([y_adl, y_fall], axis=0)
            nombres_clases = nombres_adl + nombres_fall
        elif nombre_dataset == "two_classes":
            # Construimos un dataset reducido de 9 clases (caminar + 8 caídas) a partir de ACC
            X_acc, y_acc_raw, _ = _cargar_dataset_base('acc')
            y_acc = _extraer_etiqueta_clase(y_acc_raw)  # ids originales 1..17

            # Seleccionar solo clases objetivo
            mascara = np.isin(y_acc, list(ACC_ID_TO_NEW.keys()))
            X = X_acc[mascara]
            y_filtrado = y_acc[mascara]

            # Remapear ids originales a índices consecutivos 0..8
            y_remap = np.vectorize(ACC_ID_TO_NEW.get)(y_filtrado)

            # Reemplazar estructuras de salida
            y_raw = y_remap
            nombres_clases = TWO_CLASSES_NOMBRES
            print(f"✓ Filtrado dataset ACC a {len(nombres_clases)} clases (caminar + caídas)")
            print(f"  Muestras seleccionadas: {X.shape[0]}")
            print("  Clases elegidas (id original -> nombre):")
            for orig_id in FALL_WALK_IDS:
                print(f"   - {orig_id:02d} → {_nombre_por_id_acc(orig_id)}")
        else:
            X, y_raw, nombres_clases = _cargar_dataset_base(nombre_dataset)

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

    # Convertir etiquetas al índice de clase correcto
    y_clase = _extraer_etiqueta_clase(y)

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
    Construye un modelo Convolucional 1D optimizado para Edge AI.

    Arquitectura:
    - Reshape (3, 151) → Permute (151, 3) para tensor secuencial correcto
    - Bloques Conv1D con BatchNormalization, ReLU y MaxPooling1D
    - GaussianNoise para regularización durante entrenamiento
    - Focal Loss para penalizar errores en fronteras de decisión
    - Global Average Pooling para reducir parámetros

    Args:
        input_shape (int): Número de características de entrada (453)
        num_clases (int): Número de clases de salida

    Returns:
        keras.Model: Modelo compilado con Focal Loss
    """
    print("\n[INFO] Construyendo modelo Conv1D para Edge AI...")

    model = models.Sequential([
        # Capa de entrada
        layers.Input(shape=(input_shape,)),

        # Reshape: (batch, 453) → (batch, 3, 151)
        layers.Reshape((3, 151), name='reshape_input'),

        # Permute: (batch, 3, 151) → (batch, 151, 3)
        # Esto ordena los datos secuencialmente para Conv1D: (samples, timesteps, channels)
        layers.Permute((2, 1), name='permute_to_conv'),

        # Data Augmentation: Inyectar ruido Gaussiano solo durante entrenamiento
        layers.GaussianNoise(stddev=0.01, name='gaussian_noise'),

        # Bloque Conv1D 1
        layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='conv1d_1'
        ),
        layers.BatchNormalization(name='bn_1'),
        layers.Activation('relu', name='relu_1'),
        layers.MaxPooling1D(pool_size=2, name='maxpool_1'),

        # Bloque Conv1D 2
        layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='conv1d_2'
        ),
        layers.BatchNormalization(name='bn_2'),
        layers.Activation('relu', name='relu_2'),
        layers.MaxPooling1D(pool_size=2, name='maxpool_2'),

        # Bloque Conv1D 3
        layers.Conv1D(
            filters=256,
            kernel_size=3,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='conv1d_3'
        ),
        layers.BatchNormalization(name='bn_3'),
        layers.Activation('relu', name='relu_3'),
        layers.MaxPooling1D(pool_size=2, name='maxpool_3'),

        # Global Average Pooling para reducir parámetros (crítico para Edge AI)
        layers.GlobalAveragePooling1D(name='global_avg_pool'),

        # Capa fully connected final
        layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='dense_final'
        ),
        layers.Dropout(0.3, name='dropout_final'),

        # Capa de salida con One-Hot Encoding (Softmax)
        layers.Dense(
            num_clases,
            activation='softmax',
            name='output'
        )
    ])

    # Compilación con Focal Loss para penalizar errores en fronteras complejas
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.CategoricalFocalCrossentropy(
            alpha=0.25,
            gamma=2.0,
            from_logits=False,
            label_smoothing=0.1
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy')
        ]
    )

    print("✓ Modelo Conv1D construido exitosamente")
    print("\nArquitectura del modelo:")
    model.summary()

    return model


# ============================================================================
# CALLBACK PERSONALIZADO: F1-Score Macro
# ============================================================================

class MacroF1Score(keras.callbacks.Callback):
    """
    Callback que calcula Macro F1-Score en cada época de validación.
    Permite usar F1-Score como métrica de checkpoint robusto al desbalanceo.
    """
    def __init__(self, X_val, y_val, verbose=1, restore_best=True):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.verbose = verbose
        self.restore_best = restore_best
        self.best_f1 = -np.inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Predicciones en validación
        y_pred_probs = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_val, axis=1) if len(self.y_val.shape) > 1 else self.y_val
        
        # Calcular Macro F1-Score
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        logs['val_macro_f1'] = macro_f1
        
        if self.verbose > 0:
            print(f" - val_macro_f1: {macro_f1:.4f}", end='')
        
        # Guardar mejor modelo por F1-Score
        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            self.best_epoch = epoch
            if self.restore_best:
                self.best_weights = [w.numpy().copy() for w in self.model.weights]


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def entrenar_modelo(model, X_train, y_train, X_val, y_val, epochs=100):
    """
    Entrena el modelo con callbacks para mejorar el rendimiento.

    Callbacks utilizados:
    - MacroF1Score: Monitorea F1-Score robusto al desbalanceo
    - EarlyStopping: Detiene el entrenamiento si no hay mejoría
    - ReduceLROnPlateau: Reduce learning rate si se estanca
    - ModelCheckpoint: Guarda el mejor modelo por val_loss
    - TensorBoard: Monitoreo en tiempo real

    Args:
        model: Modelo de Keras
        X_train, y_train: Datos de entrenamiento (y_train será convertido a One-Hot)
        X_val, y_val: Datos de validación (y_val será convertido a One-Hot)
        epochs: Número máximo de épocas

    Returns:
        keras.callbacks.History: Historial de entrenamiento
    """
    print("\n[INFO] Iniciando entrenamiento...")

    # Convertir etiquetas a One-Hot para trabajar con CategoricalFocalCrossentropy
    # En este punto, y_train e y_val son enteros (0..num_clases-1)
    num_clases = model.output_shape[-1]
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_clases)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=num_clases)

    # Calcular pesos de clase para balanceo dinámico
    # Usar y_train original (enteros) antes del one-hot
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    print(f"✓ Pesos de clase calculados (balanceo dinámico):")
    for cls_idx, weight in class_weight_dict.items():
        print(f"  Clase {cls_idx}: {weight:.4f}")

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

    # Checkpoint monitoreando val_loss (ya que usamos Focal Loss)
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=MODELS_DIR / f"{_nombre_base_modelo(DATASET_NAME)}_mejor_modelo.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Callback personalizado para monitorear Macro F1-Score
    macro_f1_callback = MacroF1Score(
        X_val=X_val,
        y_val=y_val_onehot,
        verbose=1,
        restore_best=False
    )

    tensorboard = callbacks.TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=1,
        write_graph=True
    )

    # Entrenamiento con class_weight para balanceo
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr, model_checkpoint, macro_f1_callback, tensorboard],
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
    - Macro F1-Score (robusto al desbalanceo)
    - Matriz de confusión

    Args:
        model: Modelo entrenado
        X_test, y_test: Datos de prueba (y_test puede ser enteros o One-Hot)
        nombres_clases: Nombres de las clases

    Returns:
        dict: Métricas de evaluación
    """
    print("\n[INFO] Evaluando modelo en conjunto de prueba...")

    # Predicciones
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1).reshape(-1)
    
    # Convertir y_test a formato entero si está en One-Hot
    if len(y_test.shape) > 1:
        y_test_class = np.argmax(y_test, axis=1).reshape(-1)
    else:
        y_test_class = np.asarray(y_test).reshape(-1)

    # Usar todas las clases conocidas, no solo las presentes en test/predicción
    nombres_clases = _normalizar_nombres_clases(nombres_clases)
    labels_full = np.arange(len(nombres_clases), dtype=int)
    nombres_reporte = [
        nombres_clases[i] if 0 <= i < len(nombres_clases) else f"clase_{i}"
        for i in labels_full
    ]

    # Métricas generales
    accuracy = accuracy_score(y_test_class, y_pred)
    precision = precision_score(y_test_class, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_class, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_class, y_pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_test_class, y_pred, average='macro', zero_division=0)

    print(f"\n{'='*50}")
    print(f"MÉTRICAS DE EVALUACIÓN")
    print(f"{'='*50}")
    print(f"Precisión (Accuracy):     {accuracy:.4f}")
    print(f"Precisión (Weighted):     {precision:.4f}")
    print(f"Recall (Weighted):        {recall:.4f}")
    print(f"F1-Score (Weighted):      {f1:.4f}")
    print(f"F1-Score (Macro):         {macro_f1:.4f}")

    # Reporte por clase
    print(f"\n{'='*50}")
    print(f"REPORTE POR CLASE")
    print(f"{'='*50}")
    reporte_clasificacion = classification_report(
        y_test_class,
        y_pred,
        labels=labels_full,
        target_names=nombres_reporte,
        zero_division=0
    )
    reporte_clasificacion_dict = classification_report(
        y_test_class,
        y_pred,
        labels=labels_full,
        target_names=nombres_reporte,
        zero_division=0,
        output_dict=True
    )
    print(reporte_clasificacion)

    # Matriz de confusión
    cm = confusion_matrix(y_test_class, y_pred, labels=labels_full)

    # Guardar métricas
    metricas = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'macro_f1_score': float(macro_f1),
        'num_muestras_test': int(len(y_test_class)),
        'clases': nombres_reporte,  # todas las clases esperadas
        'clases_presentes': list(np.unique(np.concatenate([y_test_class, y_pred])).astype(int)),
        'classification_report_text': reporte_clasificacion,
        'classification_report_dict': reporte_clasificacion_dict
    }

    return metricas, y_pred, y_test_class, cm


def extraer_info_modelo(model):
    """
    Extrae información técnica del modelo para reporte.
    """
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + "\n"))
    resumen_texto = buffer.getvalue()

    capas = []
    for idx, layer in enumerate(model.layers, start=1):
        salida = getattr(layer, "output_shape", None)
        if isinstance(salida, (tuple, list)):
            salida = str(salida)

        # Intentar obtener cantidad de unidades/neuronas de la capa
        unidades = None
        if hasattr(layer, "units"):
            unidades = int(getattr(layer, "units"))
        else:
            try:
                # Para capas sin atributo units tomamos la última dimensión del output_shape
                salida_shape = layer.output_shape
                if isinstance(salida_shape, (tuple, list)) and salida_shape:
                    last_dim = salida_shape[-1]
                    if isinstance(last_dim, int):
                        unidades = last_dim
            except Exception:
                unidades = None

        # Detectar activación si aplica
        activacion = None
        act_attr = getattr(layer, "activation", None)
        if act_attr:
            try:
                activacion = act_attr.__name__
            except Exception:
                activacion = str(act_attr)

        capas.append({
            "indice": idx,
            "nombre": layer.name,
            "tipo": layer.__class__.__name__,
            "salida": str(salida) if salida is not None else "N/D",
            "unidades": unidades if unidades is not None else "N/D",
            "activacion": activacion if activacion is not None else "N/D",
            "parametros": int(layer.count_params()),
            "entrenable": bool(layer.trainable),
        })

    optimizer = model.optimizer
    learning_rate = getattr(optimizer, "learning_rate", None)
    try:
        lr_valor = float(tf.keras.backend.get_value(learning_rate))
    except Exception:
        lr_valor = None

    info = {
        "tipo_modelo": model.__class__.__name__,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "total_capas": int(len(model.layers)),
        "capas_ocultas_aprox": int(max(len(model.layers) - 1, 0)),
        "capas": capas,
        "parametros_totales": int(model.count_params()),
        "parametros_entrenables": int(
            np.sum([np.prod(v.shape) for v in model.trainable_weights])
        ),
        "parametros_no_entrenables": int(
            np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
        ),
        "optimizador": optimizer.__class__.__name__,
        "learning_rate": lr_valor,
        "loss": str(model.loss),
        "metricas_compile": [str(m.name) for m in model.metrics],
        "summary_texto": resumen_texto,
    }
    return info


def _to_jsonable(obj):
    """
    Convierte recursivamente objetos numpy / TensorFlow a tipos serializables por json.
    """
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (numbers.Number, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Fallback: cadena representativa
    return str(obj)


def guardar_metricas(metricas, historico, modelo_nombre, model):
    """
    Guarda las métricas en un archivo JSON.

    Args:
        metricas (dict): Diccionario con métricas
        historico: Historial de entrenamiento
        modelo_nombre (str): Nombre del modelo
        model: Modelo Keras entrenado
    """
    metricas['historico'] = {
        'loss': [float(x) for x in historico.history['loss']],
        'accuracy': [float(x) for x in historico.history['accuracy']],
        'val_loss': [float(x) for x in historico.history['val_loss']],
        'val_accuracy': [float(x) for x in historico.history['val_accuracy']]
    }
    metricas['config_entrenamiento'] = {
        'dataset_name': DATASET_NAME,
        'test_size': float(TEST_SIZE),
        'validation_size': float(VALIDATION_SIZE),
        'batch_size': int(BATCH_SIZE),
        'epochs_max': int(EPOCHS),
        'learning_rate_inicial': float(LEARNING_RATE),
        'random_state': int(RANDOM_STATE),
    }
    metricas['modelo_info'] = extraer_info_modelo(model)
    if DATASET_NAME == "two_classes":
        metricas['clases_detalle'] = TWO_CLASSES_DETALLE

    ruta_metricas = MODELS_DIR / f"{modelo_nombre}_metricas.json"
    ruta_temp = ruta_metricas.with_suffix(".json.tmp")

    metricas_jsonable = _to_jsonable(metricas)
    with open(ruta_temp, 'w') as f:
        json.dump(metricas_jsonable, f, indent=4)
    ruta_temp.replace(ruta_metricas)

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
        cli_dataset = _parse_args_cli()
        env_set = "DATASET_NAME" in os.environ
        nombre_base = cli_dataset or DATASET_NAME_RAW
        global DATASET_NAME
        DATASET_NAME = _canonical_dataset_name(nombre_base)
        if cli_dataset:
            print(f"[INFO] Dataset por CLI: '{cli_dataset}' → '{DATASET_NAME}'")
        elif not env_set and DATASET_NAME_RAW == "adl_fall_multiclass":
            print("[INFO] Usando dataset por defecto 'adl_fall_multiclass'. "
                  "Para 9 clases ejecuta con --dataset entrenamiento_9_clases "
                  "o set DATASET_NAME=entrenamiento_9_clases")
        elif DATASET_NAME != DATASET_NAME_RAW:
            print(f"[INFO] Alias de dataset detectado: '{DATASET_NAME_RAW}' → '{DATASET_NAME}'")

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
        _limpiar_modelos_antiguos(DATASET_NAME)
        modelo_base = _nombre_base_modelo(DATASET_NAME)
        modelo_path = MODELS_DIR / f"{modelo_base}_modelo.keras"
        model.save(modelo_path)
        print(f"\n✓ Modelo guardado: {modelo_path}")

        guardar_metricas(metricas, history, DATASET_NAME, model)

        # 8. Visualizar
        graficar_historico(history, modelo_base)

        # 9. Guardar artefactos para el reporte
        np.save(LOGS_DIR / f"{modelo_base}_y_test.npy", y_test_eval)
        np.save(LOGS_DIR / f"{modelo_base}_y_pred.npy", y_pred)
        np.save(LOGS_DIR / f"{modelo_base}_matriz_confusion.npy", cm)

        print("\n" + "="*70)
        print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)

        return model, history, metricas, cm

    except Exception as e:
        print(f"\n✗ Error en entrenamiento: {e}")
        raise


if __name__ == "__main__":
    main()
