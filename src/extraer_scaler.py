import numpy as np
# Importamos las funciones de tu propio script
from entrenamiento import cargar_datos, preprocesar_datos

def generar_codigo_kotlin():
    # 1. Definimos el dataset que usaste para el modelo .tflite
    dataset_name = "two_classes"
    
    print(f"Calculando escalador para dataset: {dataset_name}...")
    
    # 2. Cargamos los datos crudos usando tu función
    X, y, nombres_clases = cargar_datos(dataset_name)
    
    # 3. Ejecutamos tu función de preprocesamiento para generar el scaler
    _, _, scaler = preprocesar_datos(X, y)
    
    # 4. Extraemos las matrices matemáticas
    medias = scaler.mean_
    escalas = scaler.scale_
    
    # 5. Formateamos como código fuente de Kotlin (agregando 'f' de float)
    str_medias = ", ".join([f"{x}f" for x in medias])
    str_escalas = ", ".join([f"{x}f" for x in escalas])
    
    print("\n" + "="*60)
    print("COPIA Y PEGA EL SIGUIENTE CÓDIGO EN TU ARCHIVO DataPreprocessor.kt")
    print("="*60 + "\n")
    
    print(f"private val means = floatArrayOf({str_medias})")
    print("\n")
    print(f"private val stds = floatArrayOf({str_escalas})")
    
    print("\n" + "="*60)

# Este bloque debe ir siempre hasta el final, fuera de la función
if __name__ == "__main__":
    generar_codigo_kotlin()