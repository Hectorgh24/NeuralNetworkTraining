# Red Neuronal para Clasificación de Actividades Humanas

## 📋 Descripción

Sistema de aprendizaje profundo para la clasificación de actividades humanas usando datos de acelerómetro. Utiliza redes neuronales densas con regularización para alcanzar máxima precisión en la detección de patrones de movimiento.

**Dataset:** UniMiB-SHAR (University of Milano-Bicocca Smartphone Activity Recognition)

## 🎯 Objetivos

- ✅ Clasificar actividades humanas en tiempo real
- ✅ Alcanzar alta precisión y recall balanceados
- ✅ Generar reportes profesionales con análisis detallado
- ✅ Mantener reproducibilidad en experimentos

## 📊 Características del Dataset

| Aspecto | Valor |
|--------|-------|
| **Muestras de Entrenamiento** | ~11,000 (ACC) |
| **Características por Muestra** | 453 (datos de acelerómetro) |
| **Número de Clases** | 30 (actividades diferentes) |
| **Tasa Train/Val/Test** | 60% / 20% / 20% |

### Datasets Disponibles

- **acc**: Actividades con acelerómetro (11,771 muestras)
- **adl**: Actividades de vida diaria (7,579 muestras)
- **fall**: Detección de caídas (4,192 muestras)
- **two_classes**: Clasificación binaria (variable)

## 🏗️ Arquitectura del Modelo

```
Input (453)
    ↓
Dense (256) → ReLU → BatchNorm → Dropout(0.3)
    ↓
Dense (128) → ReLU → BatchNorm → Dropout(0.3)
    ↓
Dense (64)  → ReLU → BatchNorm → Dropout(0.2)
    ↓
Dense (32)  → ReLU → Dropout(0.2)
    ↓
Dense (30)  → Softmax (Salida)
```

### Estrategias Aplicadas

- **Normalización:** StandardScaler
- **Función de Pérdida:** Sparse Categorical Crossentropy
- **Optimizador:** Adam (lr=0.001)
- **Regularización:** L2 (0.001) + Dropout + BatchNormalization
- **Callbacks:**
  - EarlyStopping (paciencia=15)
  - ReduceLROnPlateau (factor=0.5)
  - ModelCheckpoint
  - TensorBoard

## 📦 Instalación

### 1. Requisitos previos

- Python 3.8+
- pip o conda

### 2. Clonar el repositorio

```bash
git clone https://github.com/Hectorgh24/NeuralNetworkTraining.git
cd NeuralNetworkTraining
```

### 3. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Preparar datos

El proyecto espera los datos en `data/raw/`. Si solo tienes archivos `.mat`:

```bash
python convert_mat_to_npz.py
python convert_to_float32.py
```

## 🚀 Uso

### 1. Entrenar el modelo

```bash
cd src
python entrenamiento.py
```

**Salida generada:**
- `models/acc_modelo.h5` - Modelo entrenado
- `models/acc_mejor_modelo.h5` - Mejor modelo durante entrenamiento
- `models/acc_metricas.json` - Métricas de evaluación
- `logs/acc_historico.png` - Gráfico de pérdida y precisión
- `logs/acc_y_test.npy` - Etiquetas reales
- `logs/acc_y_pred.npy` - Predicciones del modelo
- `logs/acc_matriz_confusion.npy` - Matriz de confusión

### 2. Generar reporte PDF

```bash
cd src
python generar_reporte.py
```

**Salida generada:**
- `models/acc_reporte_entrenamiento.pdf` - Reporte profesional completo

El PDF incluye:
- Resumen ejecutivo
- Tabla de métricas principales
- Matriz de confusión detallada
- Análisis de desempeño
- Histórico de entrenamiento
- Información técnica

### 3. Usar el modelo entrenado

```python
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar modelo
modelo = tf.keras.models.load_model('models/acc_modelo.h5')

# Cargar y normalizar nuevos datos
X_nuevo = np.load('data/raw/acc_data.npz')['acc_data'][:10]
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X_nuevo)

# Realizar predicción
predicciones = modelo.predict(X_normalizado)
clases = np.argmax(predicciones, axis=1)
```

## 📈 Métricas de Evaluación

El sistema calcula automáticamente:

- **Precisión (Accuracy):** Porcentaje general de aciertos
- **Precisión por clase:** Qué tan bien clasifica cada actividad
- **Recall:** Cuántas muestras reales detecta de cada clase
- **F1-Score:** Balance entre precisión y recall
- **Matriz de Confusión:** Detalle de errores por clase

## 📁 Estructura del Proyecto

```
TensorFlow/
├── src/
│   ├── entrenamiento.py           # Script principal de entrenamiento
│   ├── generar_reporte.py         # Generador de reportes PDF
│   └── load_example_npz.py        # Ejemplo de carga de datos
├── data/
│   ├── raw/                       # Datos NPZ float32 (local)
│   ├── processed/                 # Datos procesados (local)
│   └── raw_float64_original/      # Backup float64 (local)
├── models/                        # Modelos entrenados y reportes (git ignored)
├── logs/                          # Logs y gráficos (git ignored)
├── convert_mat_to_npz.py          # Convertidor MAT → NumPy
├── convert_to_float32.py          # Optimizador float64 → float32
├── entrenamiento.py               # Código principal
├── requirements.txt               # Dependencias Python
└── README.md                      # Este archivo
```

## 🔧 Configuración Avanzada

### Cambiar dataset

En `src/entrenamiento.py`, línea ~51:

```python
DATASET_NAME = 'adl'  # Cambiar a 'adl', 'fall' o 'two_classes'
```

### Ajustar hiperparámetros

En `src/entrenamiento.py`, líneas ~56-63:

```python
BATCH_SIZE = 32        # Tamaño de lote
EPOCHS = 100           # Número máximo de épocas
LEARNING_RATE = 0.001  # Tasa de aprendizaje
TEST_SIZE = 0.2        # 20% para prueba
VALIDATION_SIZE = 0.2  # 20% para validación
```

### Modificar arquitectura

En `src/entrenamiento.py`, función `construir_modelo()`:

```python
# Agregar capas o cambiar tamaños
layers.Dense(512, activation='relu', ...)
```

## 📊 Resultados Esperados

Con la configuración actual, se espera:

- **Accuracy:** 85-92%
- **Precision:** 84-91%
- **Recall:** 84-91%
- **F1-Score:** 84-91%
- **Tiempo de entrenamiento:** 5-15 minutos (CPU), 1-3 minutos (GPU)

*Los valores varían según el dataset específico*

## 🐛 Solución de Problemas

### Error: "No module named 'tensorflow'"

```bash
pip install tensorflow
```

### Error: "File not found" en datos

Asegurate que los datos estén en `data/raw/` y sean archivos `.npz`

```bash
python convert_mat_to_npz.py
python convert_to_float32.py
```

### GPU no se detecta

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Si dice `[]`, configura manualmente:

```python
# En entrenamiento.py, agregar al inicio:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 0
```

## 📚 Referencias

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras API](https://keras.io/api/)
- [scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [UniMiB-SHAR Dataset](http://www.sensorlab.org/data/)

## 📝 Licencia

Este proyecto utiliza la licencia Apache 2.0. Ver archivo `LICENSE` para más detalles.

## ✍️ Autor

Desarrollado para fines educativos y de investigación en clasificación de actividades humanas.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

**Última actualización:** Marzo 2026
