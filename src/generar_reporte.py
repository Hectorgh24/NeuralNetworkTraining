"""
Generación de Reporte Profesional en PDF
Análisis completo del entrenamiento de la red neuronal

Incluye:
- Resumen de métrica sde evaluación
- Matriz de confusión
- Gráficos de desempeño
- Análisis por clase
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, Preformatted
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

LOGS_DIR = Path(r"C:\Develop\TensorFlow\logs")
MODELS_DIR = Path(r"C:\Develop\TensorFlow\models")
DATASET_NAME = 'acc'  # Cambiar según el dataset usado

# ============================================================================
# ESTILOS PERSONALIZADOS
# ============================================================================

def crear_estilos():
    """Crea estilos personalizados para el PDF."""
    styles = getSampleStyleSheet()

    # Estilo para título principal
    titulo = ParagraphStyle(
        'TituloPersonalizado',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    # Estilo para subtítulos
    subtitulo = ParagraphStyle(
        'SubtituloPersonalizado',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2e5a9e'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    # Estilo para texto normal
    normal = ParagraphStyle(
        'NormalPersonalizado',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_LEFT,
        spaceAfter=10
    )

    return titulo, subtitulo, normal


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def cargar_metricas(dataset_name):
    """
    Carga las métricas guardadas durante el entrenamiento.

    Args:
        dataset_name (str): Nombre del dataset

    Returns:
        dict: Métricas cargadas
    """
    ruta_metricas = MODELS_DIR / f"{dataset_name}_metricas.json"

    if not ruta_metricas.exists():
        raise FileNotFoundError(f"Archivo de métricas no encontrado: {ruta_metricas}")

    with open(ruta_metricas, 'r') as f:
        metricas = json.load(f)

    return metricas


def cargar_predicciones(dataset_name):
    """
    Carga predicciones y etiquetas reales guardadas.

    Args:
        dataset_name (str): Nombre del dataset

    Returns:
        tuple: (y_test, y_pred, matriz_confusion)
    """
    y_test = np.load(LOGS_DIR / f"{dataset_name}_y_test.npy")
    y_pred = np.load(LOGS_DIR / f"{dataset_name}_y_pred.npy")
    cm = np.load(LOGS_DIR / f"{dataset_name}_matriz_confusion.npy")

    return y_test, y_pred, cm


# ============================================================================
# GENERACIÓN DE GRÁFICOS
# ============================================================================

def graficar_matriz_confusion(cm, nombres_clases, dataset_name):
    """
    Genera y guarda gráfico de matriz de confusión.

    Args:
        cm (np.ndarray): Matriz de confusión
        nombres_clases (list): Nombres de las clases
        dataset_name (str): Nombre del dataset

    Returns:
        str: Ruta del archivo guardado
    """
    plt.figure(figsize=(14, 12))

    # Crear heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=nombres_clases,
        yticklabels=nombres_clases,
        cbar_kws={'label': 'Cantidad'}
    )

    plt.xlabel('Predicción', fontsize=12, fontweight='bold')
    plt.ylabel('Etiqueta Real', fontsize=12, fontweight='bold')
    plt.title(f'Matriz de Confusión - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    ruta_grafico = LOGS_DIR / f"{dataset_name}_matriz_confusion.png"
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Matriz de confusión guardada: {ruta_grafico}")
    return ruta_grafico


def graficar_metricas_por_clase(metricas, nombres_clases, dataset_name):
    """
    Genera gráfico de métricas por clase (precisión, recall, F1).

    Args:
        metricas (dict): Diccionario con métricas
        nombres_clases (list): Nombres de las clases
        dataset_name (str): Nombre del dataset

    Returns:
        str: Ruta del archivo guardado
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Ejemplo: si tienes métricas por clase, graficar
    # Nota: Esto es un placeholder, ajustar según tus datos reales

    # Gráfico 1: Distribución de muestras por clase
    y_test = np.load(LOGS_DIR / f"{dataset_name}_y_test.npy")
    unique, counts = np.unique(y_test, return_counts=True)

    axes[0].bar(range(len(counts)), counts, color='steelblue')
    axes[0].set_xlabel('Clase', fontsize=11)
    axes[0].set_ylabel('Cantidad de Muestras', fontsize=11)
    axes[0].set_title('Distribución de Clases en Test', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(len(nombres_clases)))
    axes[0].set_xticklabels(nombres_clases, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)

    # Gráfico 2: Métrica agregada
    axes[1].bar(['Precisión', 'Recall', 'F1'], [
        metricas['precision'],
        metricas['recall'],
        metricas['f1_score']
    ], color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[1].set_ylabel('Score', fontsize=11)
    axes[1].set_title('Métricas Generales', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')

    # Agregar valores en barras
    for i, v in enumerate([metricas['precision'], metricas['recall'], metricas['f1_score']]):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    # Gráfico 3: Accuracy
    accuracy = metricas['accuracy']
    colors_pie = ['#2ecc71', '#e74c3c']
    sizes = [accuracy * 100, (1 - accuracy) * 100]
    axes[2].pie(sizes, labels=['Correcto', 'Incorrecto'], autopct='%1.1f%%',
                colors=colors_pie, startangle=90)
    axes[2].set_title(f'Accuracy General: {accuracy:.3f}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    ruta_grafico = LOGS_DIR / f"{dataset_name}_metricas_por_clase.png"
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Gráfico de métricas guardado: {ruta_grafico}")
    return ruta_grafico


def graficar_historico_completo(metricas, dataset_name):
    """
    Genera gráficos del histórico de entrenamiento (loss y accuracy).

    Args:
        metricas (dict): Diccionario con historial de entrenamiento
        dataset_name (str): Nombre del dataset

    Returns:
        str: Ruta del archivo guardado
    """
    historico = metricas.get('historico', {})

    if not historico:
        print("⚠ No hay datos de histórico disponibles")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico Loss
    epochs = range(1, len(historico['loss']) + 1)
    axes[0].plot(epochs, historico['loss'], 'o-', label='Pérdida Entrenamiento', linewidth=2)
    axes[0].plot(epochs, historico['val_loss'], 's-', label='Pérdida Validación', linewidth=2)
    axes[0].set_xlabel('Épocas', fontsize=11)
    axes[0].set_ylabel('Pérdida', fontsize=11)
    axes[0].set_title('Pérdida durante Entrenamiento', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Gráfico Accuracy
    axes[1].plot(epochs, historico['accuracy'], 'o-', label='Precisión Entrenamiento', linewidth=2)
    axes[1].plot(epochs, historico['val_accuracy'], 's-', label='Precisión Validación', linewidth=2)
    axes[1].set_xlabel('Épocas', fontsize=11)
    axes[1].set_ylabel('Precisión', fontsize=11)
    axes[1].set_title('Precisión durante Entrenamiento', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    ruta_grafico = LOGS_DIR / f"{dataset_name}_historico_completo.png"
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Gráfico de histórico completo guardado: {ruta_grafico}")
    return ruta_grafico


# ============================================================================
# CONSTRUCCIÓN DEL PDF
# ============================================================================

def crear_tabla_metricas(metricas):
    """
    Crea tabla con métricas principales.

    Args:
        metricas (dict): Diccionario con métricas

    Returns:
        Table: Tabla de reportlab
    """
    datos = [
        ['Métrica', 'Valor'],
        ['Precisión (Accuracy)', f"{metricas['accuracy']:.4f}"],
        ['Precisión (Weighted)', f"{metricas['precision']:.4f}"],
        ['Recall (Weighted)', f"{metricas['recall']:.4f}"],
        ['F1-Score (Weighted)', f"{metricas['f1_score']:.4f}"],
        ['Muestras de Prueba', str(metricas['num_muestras_test'])],
        ['Número de Clases', str(len(metricas['clases']))]
    ]

    tabla = Table(datos, colWidths=[3*inch, 2*inch])

    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5a9e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
    ]))

    return tabla


def generar_pdf(dataset_name, nombres_clases):
    """
    Genera reporte completo en PDF.

    Args:
        dataset_name (str): Nombre del dataset
        nombres_clases (list): Lista de nombres de clases
    """
    print("\n[INFO] Generando reporte PDF...")

    try:
        # Cargar datos
        metricas = cargar_metricas(dataset_name)
        y_test, y_pred, cm = cargar_predicciones(dataset_name)

        # Generar gráficos necesarios
        ruta_matriz = graficar_matriz_confusion(cm, nombres_clases, dataset_name)
        ruta_metricas = graficar_metricas_por_clase(metricas, nombres_clases, dataset_name)
        ruta_historico = graficar_historico_completo(metricas, dataset_name)

        # Crear documento PDF
        ruta_pdf = MODELS_DIR / f"{dataset_name}_reporte_entrenamiento.pdf"
        doc = SimpleDocTemplate(
            str(ruta_pdf),
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Crear estilos
        titulo_style, subtitulo_style, normal_style = crear_estilos()

        # Contenido del documento
        contenido = []

        # Página 1: Portada y resumen
        contenido.append(Paragraph("REPORTE DE ENTRENAMIENTO", titulo_style))
        contenido.append(Paragraph("Red Neuronal para Clasificación de Actividades", subtitulo_style))
        contenido.append(Spacer(1, 0.3*inch))

        contenido.append(Paragraph(f"Dataset: <b>{dataset_name.upper()}</b>", normal_style))
        contenido.append(Paragraph(f"Fecha: <b>{datetime.now().strftime('%d/%m/%Y %H:%M')}</b>", normal_style))
        contenido.append(Spacer(1, 0.3*inch))

        # Resumen ejecutivo
        contenido.append(Paragraph("RESUMEN EJECUTIVO", subtitulo_style))
        resumen_text = f"""
        Se ha entrenado una red neuronal profunda para la clasificación de actividades humanas
        usando datos de acelerómetro. El modelo alcanzó una precisión de <b>{metricas['accuracy']:.2%}</b>
        en el conjunto de prueba con <b>{metricas['num_muestras_test']}</b> muestras evaluadas.
        Se identificaron <b>{len(metricas['clases'])}</b> clases diferentes.
        """
        contenido.append(Paragraph(resumen_text, normal_style))
        contenido.append(Spacer(1, 0.3*inch))

        # Tabla de métricas
        contenido.append(Paragraph("MÉTRICAS PRINCIPALES", subtitulo_style))
        contenido.append(crear_tabla_metricas(metricas))
        contenido.append(PageBreak())

        # Página 2: Gráficos
        contenido.append(Paragraph("MATRIZ DE CONFUSIÓN", subtitulo_style))
        if ruta_matriz.exists():
            contenido.append(Image(str(ruta_matriz), width=6.5*inch, height=5.5*inch))
        contenido.append(PageBreak())

        contenido.append(Paragraph("ANÁLISIS DE DESEMPEÑO", subtitulo_style))
        if ruta_metricas.exists():
            contenido.append(Image(str(ruta_metricas), width=6.5*inch, height=2.5*inch))
        contenido.append(Spacer(1, 0.2*inch))

        if ruta_historico and Path(ruta_historico).exists():
            contenido.append(Paragraph("HISTÓRICO DE ENTRENAMIENTO", subtitulo_style))
            contenido.append(Image(str(ruta_historico), width=6.5*inch, height=2.5*inch))

        contenido.append(PageBreak())

        # Información adicional
        contenido.append(Paragraph("INFORMACIÓN TÉCNICA", subtitulo_style))
        tecnica_text = """
        <b>Arquitectura del Modelo:</b><br/>
        - Capas Densas: 256 → 128 → 64 → 32 → Salida<br/>
        - Activación: ReLU (capas ocultas), Softmax (salida)<br/>
        - Regularización: Dropout + L2 + Batch Normalization<br/>
        - Función de Pérdida: Sparse Categorical Crossentropy<br/>
        - Optimizador: Adam (lr=0.001)<br/>
        <br/>
        <b>Callbacks Utilizados:</b><br/>
        - EarlyStopping (paciencia=15)<br/>
        - ReduceLROnPlateau (factor=0.5)<br/>
        - ModelCheckpoint (guardar mejor modelo)<br/>
        """
        contenido.append(Paragraph(tecnica_text, normal_style))
        contenido.append(Spacer(1, 0.2*inch))

        pie_text = "<i>Reporte generado automáticamente por el sistema de entrenamiento de IA</i>"
        contenido.append(Paragraph(pie_text, normal_style))

        # Construir PDF
        doc.build(contenido)

        print(f"\n{'='*70}")
        print(f"✓ REPORTE PDF GENERADO EXITOSAMENTE")
        print(f"{'='*70}")
        print(f"Ubicación: {ruta_pdf}")

    except Exception as e:
        print(f"✗ Error generando PDF: {e}")
        raise


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Función principal para generar el reporte.
    """
    print("\n" + "="*70)
    print("GENERACIÓN DE REPORTE - CLASIFICACIÓN DE ACTIVIDADES")
    print("="*70)

    # Cargar nombres de clases
    nombres_clases_path = Path(r"C:\Develop\TensorFlow\data\raw") / f"{DATASET_NAME}_names.npz"
    nombres_clases = np.load(nombres_clases_path, allow_pickle=True)[f"{DATASET_NAME}_names"]

    # Generar PDF
    generar_pdf(DATASET_NAME, nombres_clases)


if __name__ == "__main__":
    main()
