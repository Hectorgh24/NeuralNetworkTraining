import numpy as np
from sklearn.preprocessing import StandardScaler

def generar_codigo_kotlin_17_clases():
    # 1. Definimos el archivo del dataset original (ACC - 17 clases)
    data_file = "acc_data.npz"
    
    print(f"Calculando escalador para dataset completo de 17 clases...")
    
    try:
        # 2. Cargamos los datos crudos directamente
        X = np.load(data_file, allow_pickle=True)["acc_data"]
        
        print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características.")
        
        # 3. Ajustamos el StandardScaler al dataset completo
        scaler = StandardScaler()
        scaler.fit(X)
        
        # 4. Extraemos las matrices matemáticas
        medias = scaler.mean_
        escalas = scaler.scale_
        
        # 5. Formateamos como código fuente de Kotlin (agregando 'f' de float)
        str_medias = ", ".join([f"{x}f" for x in medias])
        str_escalas = ", ".join([f"{x}f" for x in escalas])
        
        print("\n" + "="*60)
        print("COPIA Y PEGA EL SIGUIENTE CÓDIGO EN TU ARCHIVO DataPreprocessor.kt")
        print("="*60 + "\n")
        
        print(f"private val means = floatArrayOf(\n    {str_medias}\n)")
        print("\n")
        print(f"private val stds = floatArrayOf(\n    {str_escalas}\n)")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error procesando el archivo: {e}")

if __name__ == "__main__":
    generar_codigo_kotlin_17_clases()