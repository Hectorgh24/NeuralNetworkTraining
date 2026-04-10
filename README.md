# Red Neuronal para Clasificación de Actividades Humanas

Proyecto completo para entrenar, evaluar y desplegar modelos de clasificación de actividades humanas (UniMiB-SHAR) en dispositivos móviles/edge con TensorFlow Lite.

## Tabla rápida de flujo
1) Convertir datos crudos `.mat` → `.npz` (`convert_mat_to_npz.py`, `convert_to_float32.py`).  
2) Entrenar (`src/entrenamiento.py --dataset entrenamiento_17_clases | entrenamiento_9_clases`).  
3) Generar reporte PDF (`src/generar_reporte.py --dataset ...`).  
4) Exportar a TFLite (`src/exportar_tflite.py --dataset ... [--float16]`).

## Datasets soportados
- **entrenamiento_17_clases** (alias: `adl_fall_multiclass`, `17_clases`, `acc`): actividades completas (17 clases).  
- **entrenamiento_9_clases** (alias: `two_classes`, `9_clases`): caminar + 8 tipos de caída (9 clases totales).

## Arquitectura y entrenamiento
- **Modelo:** Conv1D para Edge AI (ruido gaussiano, 3 bloques Conv1D+BN+ReLU+MaxPool, GAP, Dense 128 + Dropout, Softmax).  
- **Hiperparámetros base:** `BATCH_SIZE=32`, `EPOCHS=100`, `LR=0.001`, `TEST_SIZE=0.2`, `VAL_SIZE=0.2`.  
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard.  
- **Balanceo:** `class_weight` dinámico para caídas desbalanceadas.  
- **Entrenar:**  
  ```bash
  cd src
  python entrenamiento.py --dataset entrenamiento_17_clases
  python entrenamiento.py --dataset entrenamiento_9_clases
  ```

### Artefactos generados (por prefijo de modelo)
- Modelos Keras: `models/<prefijo>_modelo.keras`, `models/<prefijo>_mejor_modelo.keras`.
- Métricas: `models/<prefijo>_metricas.json`.
- NPY para reportes: `logs/<prefijo>_y_test.npy`, `logs/<prefijo>_y_pred.npy`, `logs/<prefijo>_matriz_confusion.npy`.
- Gráficos: `logs/<prefijo>_historico.png`.

> `<prefijo>` es `entrenamiento_17_clases` o `entrenamiento_9_clases` según dataset.

### Diferencia clave
- `_modelo.keras`: último checkpoint.
- `_mejor_modelo.keras`: mejor validación (recomendado para inferencia y TFLite).

## Datos y conversión
- Origen: archivos `.mat` de UniMiB-SHAR (acelerómetro).  
- Conversión a NPZ comprimido:  
  ```bash
  python convert_mat_to_npz.py
  python convert_to_float32.py
  ```
- Ubicación esperada: `data/raw_float32/` con `*_data.npz`, `*_labels.npz`, `*_names.npz`.

## Reportes PDF
- Script: `src/generar_reporte.py`.  
- Produce: `models/entrenamiento_17_clases.pdf` o `models/entrenamiento_9_clases.pdf`.  
- Incluye: métricas globales, por clase, matrices de confusión, histórico, tablas técnicas y configuración de entrenamiento.

## Exportación a TensorFlow Lite
- Script: `src/exportar_tflite.py` (usa el prefijo base automáticamente).  
- Ejemplos:
  ```bash
  python src/exportar_tflite.py --dataset entrenamiento_17_clases --output-dir exports/exportsTflite
  python src/exportar_tflite.py --dataset entrenamiento_9_clases --output-dir exports/exportsTflite --float16
  ```
- Salida: `exports/exportsTflite/<prefijo>_modelo.tflite`.  
- `--float16`: reduce tamaño (~40–50%) y suele mejorar latencia en hardware con soporte FP16. Úsalo cuando el dispositivo lo soporte; deja sin bandera si necesitas reproducir float32 exacto.

## Estructura del proyecto
```
TensorFlow/
├── src/
│   ├── entrenamiento.py        # Entrenamiento y guardado de métricas/modelos
│   ├── generar_reporte.py      # Reportes PDF
│   └── exportar_tflite.py      # Conversión a .tflite
├── data/                       # Datos NPZ convertidos (local)
├── models/                     # Modelos .keras y métricas .json
├── logs/                       # NPY y gráficos para reportes
├── exports/
│   ├── edge_impulse.edgei/     # Carpetas de clases para Edge Impulse
│   └── exportsTflite/          # Modelos .tflite
├── convert_mat_to_npz.py
├── convert_to_float32.py
├── requirements.txt
└── README.md
```

## Uso rápido
1. **Activar entorno**  
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Entrenar** (elige dataset): `python src/entrenamiento.py --dataset entrenamiento_17_clases`
3. **Generar PDF**: `python src/generar_reporte.py --dataset entrenamiento_17_clases`
4. **Exportar TFLite**: `python src/exportar_tflite.py --dataset entrenamiento_17_clases --output-dir exports/exportsTflite`

## Métricas esperadas (referenciales)
- Accuracy / Recall / F1 (macro): 0.84–0.92 según dataset y semilla.
- Tamaño `.keras`: ~1.9 MB; `.tflite`: ~0.16 MB (con optimización default); menos con `--float16`.

## Glosario mínimo
- **StandardScaler:** media 0, varianza 1.  
- **BatchNorm / Dropout:** estabiliza y regulariza el entrenamiento.  
- **EarlyStopping / ModelCheckpoint:** evita overfitting y guarda el mejor modelo.  
- **TFLite:** formato optimizado para inferencia en móvil/edge.

## Solución de problemas
- Falta TensorFlow: `pip install tensorflow`.  
- Datos no encontrados: coloca NPZ en `data/raw_float32/` y ejecuta los convertidores.  
- GPU no detectada: `python - <<<'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'`.

## Licencia y contribución
- Licencia MIT (ver `LICENSE`).  
- Autor: Héctor (Estudiante de Licenciatura en Tecnologías Computacionales).  
- PRs bienvenidos: fork, rama feature, PR.

---

**Última actualización:** Abril 2026
