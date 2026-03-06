# Instrucciones de Ejecución - Entrenamiento Neural

## ✅ Checklist Pre-Entrenamiento

- [ ] Python 3.8+ instalado: `python --version`
- [ ] Datos presentes: `ls data/raw/*_data.npz`
- [ ] Carpetas creadas: `models/` y `logs/`
- [ ] Dependencias instaladas: `pip install -r requirements.txt`

## 🚀 Ejecución Rápida

### 1. Test de Configuración (2-3 min)

```bash
python test_configuracion.py
```

**Esperado:** Todos los items con ✓

### 2. Entrenamiento Completo (15-30 min en CPU, 5-10 min en GPU)

```bash
cd src
python entrenamiento.py
```

**Esperado:**
- Loss disminuye progresivamente
- Accuracy aumenta
- Sin errores en consola
- Finaliza con "ENTRENAMIENTO COMPLETADO EXITOSAMENTE"

### 3. Generar Reporte PDF (1-2 min)

```bash
cd src
python generar_reporte.py
```

**Esperado:**
- Archivo PDF creado en `models/acc_reporte_entrenamiento.pdf`
- PDF contiene gráficos y métricas

## 📊 Monitoreo en Tiempo Real

### Opción 1: TensorBoard (Recomendado para ver gráficos en vivo)

```bash
# En otra terminal
tensorboard --logdir=logs
```

Luego abre en navegador: http://localhost:6006

### Opción 2: Verificar archivos generados

```bash
# Ver estado del entrenamiento
ls -lh logs/
ls -lh models/

# Ver últimas líneas del log
tail -50 logs/training.log
```

## ⚠️ Solución de Problemas

### Error: ModuleNotFoundError: No module named 'tensorflow'

```bash
pip install tensorflow
```

### Error: "No hay espacio en disco" (OOM)

Reduce tamaño de batch en `src/entrenamiento.py`:
```python
BATCH_SIZE = 16  # (en vez de 32)
```

### Error: CUDA/GPU issues

El código usa CPU por defecto. Para GPU:
```python
# Agregar en entrenamiento.py al inicio:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

### Entrenamiento muy lento

- Usa CPU: Normal, espera 15-30 min
- Reduce EPOCHS a 50 para test rápido
- Reduce dataset en `convert_to_float32.py`

## 📁 Archivos Esperados Después

```
✓ models/acc_modelo.h5
✓ models/acc_mejor_modelo.h5
✓ models/acc_metricas.json
✓ models/acc_reporte_entrenamiento.pdf

✓ logs/acc_historico.png
✓ logs/acc_y_test.npy
✓ logs/acc_y_pred.npy
✓ logs/acc_matriz_confusion.npy
```

## 🎯 Métricas que Esperar

| Métrica | Valor Esperado |
|---------|----------------|
| Accuracy | 80-92% |
| Precision | 80-91% |
| Recall | 80-91% |
| F1-Score | 80-91% |

*Varían según dataset*

## 💡 Tips

1. **Primera vez es lenta:** Descarga pesos pre-entrenados
2. **GPU acelera 5-10x:** Instala CUDA si tienes GPU
3. **Cambia dataset:** Modifica `DATASET_NAME` en entrenamiento.py
4. **Personaliza modelo:** Edita función `construir_modelo()`

## 📞 Comandos útiles

```bash
# Ver si Python puede usar GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Ver versión de TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Limpiar archivos temporales
rm -rf logs/* models/*  # (cuidado, borra entrenamiento previo)

# Ver tamaño de modelo
du -sh models/
```

## ✅ Flujo Completo Ejemplo

```bash
# 1. Instalar
pip install -r requirements.txt

# 2. Test
python test_configuracion.py
# → Ver: "TODOS LOS TEST PASARON CORRECTAMENTE"

# 3. Entrenar
cd src
python entrenamiento.py
# → Ver: "ENTRENAMIENTO COMPLETADO EXITOSAMENTE"
# → Esperar 15-30 minutos

# 4. Reporte
python generar_reporte.py
# → Ver: "REPORTE PDF GENERADO EXITOSAMENTE"

# 5. Revisar
# → Abrir models/acc_reporte_entrenamiento.pdf
```

¡Listo! 🎉
