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
- Modelos optimizados para desplegarse en **aplicaciones móviles (Kotlin/Android) mediante TFLite**.
- Dos configuraciones de clases: 17 (completo) y 9 (caminar + caídas).

## 🧱 Diferencias de Arquitectura (antes vs ahora)
- **Antes (MLP denso)**  
  - Aplanaba las 453 features → perdía la relación temporal de las 151 muestras.  
  - Más parámetros y menos eficiencia en despliegue móvil.  
- **Ahora (Conv1D)**  
  - Mantiene la forma (tiempo × ejes) y extrae patrones locales.  
  - BatchNorm + Dropout + GaussianNoise para robustez.  
  - GlobalAveragePooling reduce parámetros antes de la capa final.  
  - Mejor ajuste para exportar a TFLite y ejecutar en Android.

## 🗂️ Datasets Soportados
- **entrenamiento_17_clases** (alias: `adl_fall_multiclass`, `17_clases`, `acc`)
- **entrenamiento_9_clases** (alias: `two_classes`, `9_clases`)

## 🛠️ Pipeline Secuencial Paso a Paso (ambos modelos: 9 y 17 clases)
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
6) **Usar en aplicación móvil (Kotlin/Android)**
   - Copia el `.tflite` correspondiente (9 o 17 clases) y cárgalo con el intérprete de TensorFlow Lite.  
   - Usa NNAPI/GPU delegate si el dispositivo lo soporta; si no, el intérprete CPU funciona.

## 📦 Artefactos que se generan
- Modelos Keras: `models/<prefijo>_modelo.keras`, `models/<prefijo>_mejor_modelo.keras`
- Métricas: `models/<prefijo>_metricas.json`
- Datos para reporte: `logs/<prefijo>_y_test.npy`, `logs/<prefijo>_y_pred.npy`, `logs/<prefijo>_matriz_confusion.npy`
- Gráficos: `logs/<prefijo>_historico.png`, `logs/<prefijo>_metricas_*.png`
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
├── src/
│   ├── entrenamiento.py        # Entrenamiento y guardado de modelos/métricas
│   ├── generar_reporte.py      # Reportes PDF por dataset
│   └── exportar_tflite.py      # Conversión .keras → .tflite
├── data/
│   └── raw_float32/            # NPZ convertidos (acc/adl/fall/two_classes)
├── models/                     # .keras, métricas .json, PDFs
├── logs/                       # y_test/y_pred/matriz_confusión + gráficos
├── exports/
│   ├── edge_impulse.edgei/     # Carpetas de clases (insumos etiquetados)
│   └── exportsTflite/          # Modelos .tflite listos para Android
├── convert_mat_to_npz.py       # .mat → .npz
├── convert_to_float32.py       # Normalización a float32
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
- TensorFlow: https://www.tensorflow.org/api_docs
- Keras: https://keras.io/api/
- TensorFlow Lite Converter: https://www.tensorflow.org/lite/convert
- scikit-learn: https://scikit-learn.org/stable/
- UniMiB-SHAR dataset: https://www.dropbox.com/scl/fi/g5ig8nw9qqd253dz8woax/UniMiB-SHAR.zip?e=4&file_subpath=%2FUniMiB-SHAR%2Fdata%2Fresults%2Fresults51&rlkey=o0ltu8ivrr9rsfvdhr1bjv3cc&dl=0
- ReportLab: https://www.reportlab.com/documentation/
- Matplotlib: https://matplotlib.org/stable/contents.html
- Seaborn: https://seaborn.pydata.org/

## 📖 Glosario

## 📜 Licencia y contribución
- Licencia: **MIT** (ver `LICENSE`).
- Autor: Héctor (Estudiante de Licenciatura en Tecnologías Computacionales).
- PRs bienvenidos: fork, rama feature, PR.

---

**Última actualización:** Abril 2026
