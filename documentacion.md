# Documentación técnica: flujo completo de datos, entrenamiento y despliegue en TFLite

Este documento describe de forma secuencial el pipeline del proyecto, las decisiones de modelado y los artefactos generados, desde la conversión de datos crudos hasta la exportación de modelos para dispositivos móviles.

## 1) Datos de origen y conversión a NPZ
- **Origen**: archivos `.mat` (formato MATLAB) del dataset UniMiB-SHAR (acelerómetro).
- **Conversión**: scripts auxiliares (`convert_mat_to_npz.py`, `convert_to_float32.py`) transformaron los `.mat` a `.npz` (formato comprimido de NumPy) para acelerar carga y reducir dependencias.
- **Ubicación final**: `data/raw/` contiene `acc_data.npz`, `acc_labels.npz`, `acc_names.npz`, y equivalentes para `adl`, `fall`, `two_classes`, etc.
- **Conceptos claves**:
  - **NPZ**: contenedor zip de arrays NumPy; rápido de cargar con `np.load`.
  - **Etiquetas**: en `*_labels.npz` suelen venir como enteros (id de actividad) o one-hot; se normalizan más adelante.
  - **Nombres de clase**: `*_names.npz` trae las descripciones de cada etiqueta.

## 2) Preparación del dataset dentro del entrenamiento
- **Script**: `src/entrenamiento.py`.
- **Selección de dataset**: variable `DATASET_NAME` (CLI `--dataset` o env). Alias soportados:
  - 9 clases (caminar + 8 caídas): `entrenamiento_9_clases` / `two_classes` / `9_clases`.
  - 17 clases: `entrenamiento_17_clases` / `adl_fall_multiclass` / `17_clases`.
- **Carga**: función `cargar_datos` lee los NPZ, extrae etiquetas y normaliza nombres.
- **Filtrado 9 clases**: se toma `acc` y se enmascaran ids 3 (walking) y 10–17 (caídas); se remapean a índices 0–8.
- **Normalización de features**: `StandardScaler` convierte a media 0, varianza 1; salida `float32`.
- **División**: `train/val/test` estratificado (80/20 con 20% del train para val).

## 3) Modelo y entrenamiento
- **Arquitectura**: perceptrón multicapa (MLP) denso. No es una CNN ni un modelo “tiny”.
  - Capas densas: 256 → 128 → 64 → 32 (ReLU).
  - Regularización: L2, Batch Normalization y Dropout en bloques intermedios.
  - Capa de salida: Dense `softmax` con `num_clases`.
- **Por qué un MLP denso**:
  - Las señales del acelerómetro se usan como vector de características ya alineado (453 valores). Un MLP es suficiente para capturar relaciones globales sin costes de convoluciones ni arquitecturas secuenciales.
  - Permite un balance razonable entre capacidad y tamaño (≈1.6e5 parámetros) sin depender de kernels 1D/2D.
- **Por qué no es un modelo “tiny”**:
  - No se aplicaron reducciones agresivas de parámetros ni técnicas de compresión en el grafo. Un tiny model suele diseñarse con muchas menos unidades/capas, o con arquitecturas especializadas (p. ej., MicroNets, TinyML con depthwise separable convs, pruning o distillation).
  - Ventajas del MLP actual: entrenamiento más simple, buena precisión en los datos presentes, pipeline directo a TFLite.
  - Desventajas frente a un tiny: mayor tamaño y consumo que un modelo optimizado micro/edge; menor eficiencia en microcontroladores sin aceleradores.
- **Hiperparámetros**: `BATCH_SIZE=32`, `EPOCHS=100`, `LEARNING_RATE=0.001`.
- **Pérdida y métrica**: `sparse_categorical_crossentropy`; métrica principal `accuracy`.
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard.
- **Artefactos guardados**:
  - Modelos `.keras`:
    - `<prefijo>_modelo.keras`: estado de la última época entrenada.
    - `<prefijo>_mejor_modelo.keras`: mejor checkpoint según `val_accuracy`/`val_loss` (recomendado para despliegue).
  - Métricas JSON: `models/<dataset>_metricas.json`.
  - NPY para reporte: en `logs/` (`<prefijo>_y_test.npy`, `<prefijo>_y_pred.npy`, `<prefijo>_matriz_confusion.npy`).
  - Gráfico histórico: `logs/<prefijo>_historico.png`.

### Diferencia entre `<prefijo>_modelo.keras` y `<prefijo>_mejor_modelo.keras`
- **`_modelo.keras`**: último estado del entrenamiento; puede no coincidir con la mejor métrica de validación.
- **`_mejor_modelo.keras`**: guardado automáticamente cuando la métrica de validación mejora. Es el candidato principal para producción y conversión a TFLite porque maximiza la generalización observada.

## 4) Generación de reporte en PDF
- **Script**: `src/generar_reporte.py`.
- **Entrada**: mismo `DATASET_NAME`/`--dataset` que en entrenamiento.
- **Salida**: `models/entrenamiento_9_clases.pdf` o `entrenamiento_17_clases.pdf`.
- **Incluye**: métricas globales, métricas por clase, matrices de confusión (conteo y normalizada), histórico de loss/acc, tablas técnicas (config, capas, parámetros).
- **Fuentes**: lee métricas JSON y NPY generados en entrenamiento.

## 5) Exportación a TensorFlow Lite (TFLite)
- **Script**: `src/exportar_tflite.py`.
- **Cómo funciona**:
  1. Carga el modelo `.keras` (preferentemente el `_mejor_modelo.keras`).
  2. Construye un convertidor `tf.lite.TFLiteConverter.from_keras_model`.
  3. (Opcional) Aplica cuantización `float16` a los pesos si se indica `--float16`.
  4. Genera el archivo `.tflite` en `exports/tflite/` con el mismo prefijo.
- **Uso típico**:
  - 9 clases: `python src/exportar_tflite.py --dataset entrenamiento_9_clases --float16`
  - 17 clases: `python src/exportar_tflite.py --dataset entrenamiento_17_clases`
  - Ruta directa: `python src/exportar_tflite.py --input models/entrenamiento_9_clases_mejor_modelo.keras`
- **¿Para qué sirve TFLite?**
  - Es el formato optimizado de TensorFlow para inferencia en móvil/edge. Reduce tamaño y depende de kernels optimizados (NNAPI, GPU delegate, XNNPACK) para mejorar latencia y consumo.

## 6) Resumen rápido de extensiones y para qué sirven
- `.npz`: datos y etiquetas preprocesadas (entrada al entrenamiento).
-.keras`: modelo entrenado (mejor usar el `_mejor_modelo` para despliegue).
- `.json`: métricas detalladas del entrenamiento.
- `.npy`: predicciones, etiquetas y matriz de confusión para reportes.
- `.png`: visualizaciones (histórico, métricas, matrices).
- `.pdf`: reporte legible de resultados.
- `.tflite`: modelo optimizado para móvil.

## 7) Glosario breve
- **StandardScaler**: normaliza features a media 0, varianza 1.
- **BatchNorm**: estabiliza activaciones durante entrenamiento.
- **Dropout**: apaga aleatoriamente neuronas para reducir overfitting.
- **EarlyStopping**: detiene antes si no mejora `val_loss`.
- **ModelCheckpoint**: guarda el mejor modelo observado.
- **ReduceLROnPlateau**: baja el learning rate cuando se estanca.
- **Sparse Categorical Crossentropy**: pérdida para clasificación multiclase con etiquetas enteras.

## 8) Flujo sugerido de uso
1. (Opcional) Re-entrenar con el dataset deseado:
   - `set DATASET_NAME=entrenamiento_9_clases` o `--dataset entrenamiento_9_clases`
   - `python src/entrenamiento.py`
2. Generar reporte PDF:
   - `python src/generar_reporte.py --dataset entrenamiento_9_clases`
3. Exportar a TFLite:
   - `python src/exportar_tflite.py --dataset entrenamiento_9_clases --float16`
4. Llevar el `.tflite` a la app móvil y cargarlo con TensorFlow Lite (Android/iOS).

## 9) Cuándo usar `--float16` al exportar a TFLite
- **Qué hace**: aplica cuantización a float16 en los pesos durante la conversión. Reduce tamaño y, en muchos dispositivos, mejora latencia.
- **Ventajas**:
  - Tamaño menor del archivo `.tflite` (≈40–50% menos que float32).
  - Mejor aprovechamiento en GPUs/NNAPI que soportan float16; suele dar inferencia más rápida.
  - Preserva más precisión que int8 cuando no se ha preparado un pipeline de quantization-aware training.
- **Desventajas / riesgos**:
  - Ligerísima pérdida de precisión numérica frente a float32 (normalmente muy pequeña).
  - Beneficio en CPU puede ser limitado si el hardware no acelera float16; en ese caso el tamaño sí baja, pero la velocidad no siempre.
- **Cuándo activarlo**:
  - Objetivo móvil/edge y quieres reducir tamaño sin rehacer el entrenamiento.
  - Tu dispositivo soporta hardware float16 (GPUs móviles modernas, NNAPI en Android).
- **Cuándo no activarlo**:
  - Necesitas reproducir exactamente los resultados float32 y el tamaño no es crítico.
  - Vas a aplicar una cuantización int8 calibrada (requeriría pasos adicionales no implementados aquí).

### Aclaración de conceptos: float32, float16, int8 y quantization-aware training
- **float32**: formato de 32 bits por número (precisión estándar en entrenamiento e inferencia). Mayor precisión y mayor tamaño de modelo.
- **float16**: 16 bits por número. Menor tamaño, ligera pérdida de precisión. Útil cuando el hardware lo acelera.
- **int8**: 8 bits por número. Reduce fuertemente el tamaño y suele acelerar CPU/edge, pero requiere calibración o entrenamiento específico para minimizar la caída de precisión.
- **Quantization-aware training (QAT)**: técnica de entrenamiento donde se simula la cuantización (típicamente int8) durante el entrenamiento para que el modelo “aprenda” a ser robusto al redondeo. No se aplicó en este proyecto; usamos post-training quantization float16, que es más sencilla y suele mantener buena precisión sin reentrenar.
