# Red Neuronal para Clasificación de Actividades Humanas

Proyecto completo para entrenar, evaluar y desplegar modelos de clasificación de actividades humanas (UniMiB-SHAR) en dispositivos móviles/edge con TensorFlow Lite.  
Autor: **Héctor (Estudiante de Licenciatura en Tecnologías Computacionales)**.

## 🧭 Índice
1. [Visión General](#-visión-general)
2. [Diferencias de Arquitectura (antes vs ahora)](#-diferencias-de-arquitectura-antes-vs-ahora)
3. [Datasets Soportados](#-datasets-soportados)
4. [Pipeline Secuencial Paso a Paso](#-pipeline-secuencial-paso-a-paso)
5. [Artefactos que se generan](#-artefactos-que-se-generan)
6. [Exportación a TensorFlow Lite](#-exportación-a-tensorflow-lite)
7. [Estructura del Proyecto](#-estructura-del-proyecto)
8. [Errores Comunes y cómo resolverlos](#-errores-comunes-y-cómo-resolverlos)
9. [Documentación usada](#-documentación-usada)
10. [Glosario ampliado](#-glosario-ampliado)
11. [Licencia y contribución](#-licencia-y-contribución)

## 📝 Visión General
- Clasificación de actividades humanas con acelerómetro (UniMiB-SHAR).
- Enfoque orientado a **Edge AI**: modelos ligeros + exportación TFLite.
- Dos configuraciones de clases: 17 (completo) y 9 (caminar + caídas).

## 🧱 Diferencias de Arquitectura (antes vs ahora)
- **Antes (MLP denso):**
  - Entrada aplanada (453 valores) sin respetar la estructura temporal.
  - Menos adecuado para capturar patrones locales de movimiento.
  - Tamaño mayor y menos eficiente en dispositivos edge.
- **Ahora (Conv1D):**
  - Reordena datos a (tiempo, ejes) y aplica convoluciones 1D en ventanas cortas.
  - BatchNorm + Dropout + GaussianNoise para robustez y regularización.
  - GlobalAveragePooling para reducir parámetros antes de la capa densa final.
  - Mejor generalización para secuencias de acelerómetro y menor huella para TFLite.
> Si no estás seguro de cifras exactas, quédate con la idea: Conv1D captura patrones temporales y suele ser más liviano que el MLP previo.

## 🗂️ Datasets Soportados
- **entrenamiento_17_clases** (alias: `adl_fall_multiclass`, `17_clases`, `acc`)
- **entrenamiento_9_clases** (alias: `two_classes`, `9_clases`)

## 🛠️ Pipeline Secuencial Paso a Paso
1) **Preparar entorno**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2) **Convertir datos crudos** (`.mat` → `.npz`, una sola vez)
   ```bash
   python convert_mat_to_npz.py
   python convert_to_float32.py
   ```
   - Salida esperada: `data/raw_float32/*_data.npz`, `*_labels.npz`, `*_names.npz`.
3) **Entrenar** (elige dataset)
   ```bash
   python src/entrenamiento.py --dataset entrenamiento_17_clases
   python src/entrenamiento.py --dataset entrenamiento_9_clases
   ```
4) **Generar reporte PDF**
   ```bash
   python src/generar_reporte.py --dataset entrenamiento_17_clases
   python src/generar_reporte.py --dataset entrenamiento_9_clases
   ```
5) **Exportar a TFLite**
   ```bash
   python src/exportar_tflite.py --dataset entrenamiento_17_clases --output-dir exports/exportsTflite
   python src/exportar_tflite.py --dataset entrenamiento_9_clases --output-dir exports/exportsTflite --float16
   ```
6) **Usar en móvil/edge**
   - Copia el `.tflite` a tu app y cárgalo con TensorFlow Lite (NNAPI / GPU delegate si aplica).

## 📦 Artefactos que se generan
- Modelos: `models/<prefijo>_modelo.keras`, `models/<prefijo>_mejor_modelo.keras`
- Métricas: `models/<prefijo>_metricas.json`
- Datos para reporte: `logs/<prefijo>_y_test.npy`, `logs/<prefijo>_y_pred.npy`, `logs/<prefijo>_matriz_confusion.npy`
- Gráficos: `logs/<prefijo>_historico.png`, métricas y matrices
- Reporte: `models/<prefijo>.pdf`
- TFLite: `exports/exportsTflite/<prefijo>_modelo.tflite`
> `<prefijo>` = `entrenamiento_17_clases` o `entrenamiento_9_clases`.

## 📲 Exportación a TensorFlow Lite
- Script: `src/exportar_tflite.py`
- `--float16` (opcional): reduce tamaño; úsalo si el dispositivo soporta FP16.  
- Si no especificas `--input`, toma automáticamente el modelo .keras según el dataset.

## 🗺️ Estructura del Proyecto
```
TensorFlow/
├── src/                     # Código de entrenamiento, reportes y exportación
├── data/                    # Datos NPZ convertidos (local)
├── models/                  # Modelos .keras, métricas .json, PDFs
├── logs/                    # NPY y gráficos
├── exports/
│   ├── edge_impulse.edgei/  # Carpetas de clases para Edge Impulse
│   └── exportsTflite/       # Modelos .tflite
├── convert_mat_to_npz.py
├── convert_to_float32.py
├── requirements.txt
└── README.md
```

## 🐞 Errores Comunes y cómo resolverlos
- **Falta TensorFlow / dependencias**: `pip install -r requirements.txt`.
- **No encuentra datos (`File not found`)**: verifica que `data/raw_float32/` contenga los `.npz`; si no, ejecuta los convertidores.
- **GPU no aparece**: `python - <<<'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'`; si retorna `[]`, revisa drivers/CUDA o usa CPU.
- **ValueError por shapes**: asegura que uses el dataset correcto; limpia artefactos viejos borrando `logs/*.npy` y `models/*_metricas.json` si cambiaste de configuración.
- **Reporte falla al cargar métricas**: confirma que corriste entrenamiento antes del PDF y que el prefijo coincide (`entrenamiento_9_clases` vs `entrenamiento_17_clases`).

## 📚 Documentación usada
- TensorFlow / Keras API (modelado, callbacks, TFLite Converter)
- scikit-learn (StandardScaler, class_weight, métricas)
- UniMiB-SHAR dataset (especificación de actividades y archivos .mat)
- ReportLab + Matplotlib + Seaborn (generación de PDF y gráficos)

## 📖 Glosario ampliado
- **StandardScaler**: normaliza características a media 0, varianza 1.
- **BatchNormalization**: estabiliza activaciones para entrenar más rápido y estable.
- **Dropout**: apaga neuronas aleatoriamente durante el entrenamiento para reducir sobreajuste.
- **GaussianNoise**: añade ruido controlado a la entrada para robustez.
- **EarlyStopping**: detiene el entrenamiento si la métrica de validación deja de mejorar.
- **ModelCheckpoint**: guarda el mejor modelo observado en validación.
- **ReduceLROnPlateau**: baja la tasa de aprendizaje cuando no mejora la validación.
- **Class Weighting**: pesos por clase para compensar desbalance.
- **Focal Loss**: pérdida que penaliza más los ejemplos difíciles/clases minoritarias.
- **Conv1D**: convoluciones sobre secuencias unidimensionales (tiempo); detectan patrones locales.
- **GlobalAveragePooling**: resume cada mapa de características promediando, reduciendo parámetros antes de la capa densa.
- **TFLite**: formato ligero de TensorFlow para inferencia en móvil/edge.
- **Cuantización Float16**: convierte pesos a FP16; reduce tamaño y puede acelerar en hardware compatible.
- **NNAPI / GPU delegate**: aceleradores de TFLite en Android (CPU/GPU/DSP/TPU según dispositivo).
- **Numpy NPZ**: contenedor comprimido de arrays; rápido de cargar.
- **Matriz de confusión**: tabla de aciertos/errores por clase.
- **F1-score**: media armónica de precisión y recall; útil en desbalance.

## 📜 Licencia y contribución
- Licencia: **MIT** (ver `LICENSE`).
- Autor: Héctor (Estudiante de Licenciatura en Tecnologías Computacionales).
- PRs bienvenidos: fork, rama feature, PR.

---

**Última actualización:** Abril 2026
