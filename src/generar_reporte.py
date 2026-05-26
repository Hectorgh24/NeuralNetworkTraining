"""
Generación de Reporte Profesional en PDF
Análisis completo del entrenamiento de la red neuronal convolucional (Conv1D)

Arquitectura:
- Red Convolucional 1D (Conv1D) optimizada para Edge AI
- Bloques Conv1D + BatchNormalization + ReLU + MaxPooling1D
- Global Average Pooling para reducir parámetros
- Focal Loss con Label Smoothing para fronteras complejas
- GaussianNoise como regularización en entrenamiento

Incluye:
- Resumen de métricas de evaluación
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
    Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import os
import sys

# Evita fallos de impresión en consolas Windows con codificación cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
REPORTES_DIR = BASE_DIR / "reportes"
DATASET_NAME_RAW = os.getenv("DATASET_NAME", "adl_fall_multiclass").strip()
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
# UTILIDADES
# ============================================================================

def _get_reporte_dir(dataset_name):
    """
    Crea y devuelve la ruta de la subcarpeta dedicada para el reporte.
    """
    base = _nombre_base_modelo(dataset_name)
    ruta = REPORTES_DIR / f"reporte_{base}"
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta

def _borrar_artefactos_previos(dataset_name):
    """
    Elimina artefactos gráficos y PDF previos para garantizar regeneración fresca.
    """
    base = _nombre_base_modelo(dataset_name)
    pdf_nombre = _nombre_pdf(dataset_name)
    patrones = [
        f"{base}_matriz_confusion_abs.png",
        f"{base}_matriz_confusion_norm.png",
        f"{base}_metricas_globales.png",
        f"{base}_metricas_por_clase.png",
        f"{base}_soporte_por_clase.png",
        f"{base}_historico_completo.png",
        f"{base}_reporte_entrenamiento.pdf",
        pdf_nombre,
    ]
    reporte_dir = _get_reporte_dir(dataset_name)
    for nombre in patrones:
        rutas = [
            LOGS_DIR / nombre,
            MODELS_DIR / nombre,
            reporte_dir / nombre
        ]
        for ruta in rutas:
            try:
                ruta.unlink(missing_ok=True)
            except Exception:
                pass


def _nombre_pdf(dataset_name):
    """
    Devuelve el nombre de archivo PDF deseado para cada dataset.
    """
    if dataset_name == "two_classes":
        return "entrenamiento_9_clases.pdf"
    if dataset_name in ("adl_fall_multiclass", "acc"):
        return "entrenamiento_17_clases.pdf"
    return f"{dataset_name}_reporte_entrenamiento.pdf"


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


def _nombre_base_modelo(dataset_name):
    """
    Devuelve el prefijo base para artefactos (coincide con entrenamiento.py).
    """
    return MODEL_BASE_BY_DATASET.get(dataset_name, dataset_name)


def _parse_args_cli():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", "--data", dest="dataset", default=None)
    args, _ = parser.parse_known_args()
    return args.dataset

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
    base = _nombre_base_modelo(dataset_name)
    ruta_metricas = MODELS_DIR / f"{base}_metricas.json"

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
    base = _nombre_base_modelo(dataset_name)
    y_test = np.load(LOGS_DIR / f"{base}_y_test.npy")
    y_pred = np.load(LOGS_DIR / f"{base}_y_pred.npy")
    cm = np.load(LOGS_DIR / f"{base}_matriz_confusion.npy")

    return y_test, y_pred, cm


# ============================================================================
# GENERACIÓN DE GRÁFICOS
# ============================================================================

def graficar_matriz_confusion(cm, nombres_clases, dataset_name):
    """
    Genera y guarda gráfico de matriz de confusión mejorado con alta resolución.
    
    Mejoras:
    - Genera matrices en imágenes INDEPENDIENTES (abs y norm) para maximizar el tamaño en el PDF.
    - Uso de abreviaciones lógicas para 17 clases con leyenda explicativa.
    - Títulos descriptivos para dejar claro qué mide cada matriz.
    - Alta calidad (600 DPI) para evitar pixelado.

    Args:
        cm (np.ndarray): Matriz de confusión
        nombres_clases (list): Nombres de las clases
        dataset_name (str): Nombre del dataset

    Returns:
        tuple: (ruta_abs, ruta_norm) Rutas de los archivos guardados
    """
    num_clases = max(1, len(nombres_clases))
    
    # Abreviaciones lógicas para dataset de muchas clases (ej. 17 clases)
    usar_abreviaturas = num_clases > 10
    etiquetas_ejes = nombres_clases
    texto_leyenda = ""
    
    if usar_abreviaturas:
        abreviaciones = []
        leyenda_items = []
        for nombre in nombres_clases:
            partes = nombre.replace('_', ' ').replace('-', ' ').split()
            if len(partes) == 1:
                abrev = nombre[:3].upper()
            else:
                abrev = "".join([p[:2].capitalize() for p in partes])
            
            # Resolver colisiones si se generan dos abreviaciones iguales
            orig = abrev
            c = 1
            while abrev in abreviaciones:
                abrev = f"{orig}{c}"
                c += 1
            abreviaciones.append(abrev)
            leyenda_items.append(f"{abrev}: {nombre}")
            
        etiquetas_ejes = abreviaciones
        
        # Construir texto de leyenda dividido en múltiples líneas
        lineas_leyenda = []
        n_por_linea = 5 if num_clases <= 15 else 6
        for i in range(0, len(leyenda_items), n_por_linea):
            lineas_leyenda.append(" | ".join(leyenda_items[i:i+n_por_linea]))
        texto_leyenda = "Leyenda de Abreviaciones:\n" + "\n".join(lineas_leyenda)
    
    # Figsize adaptado para UNA sola matriz por figura
    if num_clases > 12:
        figsize = (10, 8) if usar_abreviaturas else (12, 10)
    elif num_clases > 8:
        figsize = (9, 7)
    else:
        figsize = (8, 6)
    
    # Habilitar alta resolución en Matplotlib
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    
    mostrar_anotaciones = True

    # Tamaño de fuente dinámico
    if num_clases > 15:
        annot_fontsize = 8
    elif num_clases > 10:
        annot_fontsize = 9
    else:
        annot_fontsize = 11

    base = _nombre_base_modelo(dataset_name)
    reporte_dir = _get_reporte_dir(dataset_name)
    ruta_abs = reporte_dir / f"{base}_matriz_confusion_abs.png"
    ruta_norm = reporte_dir / f"{base}_matriz_confusion_norm.png"

    # =========================================================
    # 1. Matriz en conteos absolutos
    # =========================================================
    # Corregir problema de dimensiones si cm es más grande que las clases reales (ej. padding a 34x34)
    cm = cm[:num_clases, :num_clases]
    
    fig_abs, ax_abs = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=mostrar_anotaciones,
        fmt='d',
        cmap='Blues',
        xticklabels=etiquetas_ejes,
        yticklabels=etiquetas_ejes,
        cbar_kws={'label': 'Cantidad de Muestras'},
        annot_kws={'size': annot_fontsize},
        linewidths=0.5,
        linecolor='lightgray',
        square=True,
        ax=ax_abs
    )
    ax_abs.set_title('Matriz de Confusión Absoluta\n(Total de muestras: Real vs Predicción)', fontsize=14, fontweight='bold', pad=15)
    ax_abs.set_xlabel('Clase Predicha por el Modelo', fontsize=12, fontweight='bold')
    ax_abs.set_ylabel('Clase Real (Verdadera)', fontsize=12, fontweight='bold')
    ax_abs.set_xticklabels(ax_abs.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax_abs.set_yticklabels(ax_abs.get_yticklabels(), rotation=0, fontsize=10)

    fig_abs.tight_layout()
    
    if usar_abreviaturas:
        plt.subplots_adjust(bottom=0.25)
        fig_abs.text(0.5, 0.02, texto_leyenda, ha='center', va='bottom', fontsize=10, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.9))

    fig_abs.savefig(ruta_abs, dpi=600, bbox_inches='tight', format='png')
    plt.close(fig_abs)

    # =========================================================
    # 2. Matriz normalizada por fila (Sensibilidad / Recall)
    # =========================================================
    cm_f = cm.astype(np.float32)
    suma_filas = cm_f.sum(axis=1, keepdims=True)
    suma_filas[suma_filas == 0] = 1.0
    cm_norm = cm_f / suma_filas
    
    fig_norm, ax_norm = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=mostrar_anotaciones,
        fmt='.2f',
        cmap='Greens',
        xticklabels=etiquetas_ejes,
        yticklabels=etiquetas_ejes,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Proporción de Aciertos'},
        annot_kws={'size': annot_fontsize},
        linewidths=0.5,
        linecolor='lightgray',
        square=True,
        ax=ax_norm
    )
    ax_norm.set_title('Matriz de Confusión Normalizada\n(Sensibilidad: % de aciertos por clase real)', fontsize=14, fontweight='bold', pad=15)
    ax_norm.set_xlabel('Clase Predicha por el Modelo', fontsize=12, fontweight='bold')
    ax_norm.set_ylabel('Clase Real (Verdadera)', fontsize=12, fontweight='bold')
    ax_norm.set_xticklabels(ax_norm.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax_norm.set_yticklabels(ax_norm.get_yticklabels(), rotation=0, fontsize=10)

    fig_norm.tight_layout()
    
    if usar_abreviaturas:
        plt.subplots_adjust(bottom=0.25)
        fig_norm.text(0.5, 0.02, texto_leyenda, ha='center', va='bottom', fontsize=10, 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.9))

    fig_norm.savefig(ruta_norm, dpi=600, bbox_inches='tight', format='png')
    plt.close(fig_norm)

    print("✓ Matrices de confusión (abs y norm) guardadas individualmente.")
    return ruta_abs, ruta_norm


def _obtener_metricas_por_clase(metricas):
    """
    Extrae precisión, recall, F1 y soporte por clase desde classification_report_dict.
    """
    reporte_dict = metricas.get('classification_report_dict', {})
    clases = metricas.get('clases', [])
    filas = []
    for clase in clases:
        fila = reporte_dict.get(clase, {})
        if isinstance(fila, dict):
            filas.append({
                'clase': clase,
                'precision': float(fila.get('precision', 0.0)),
                'recall': float(fila.get('recall', 0.0)),
                'f1': float(fila.get('f1-score', 0.0)),
                'support': int(fila.get('support', 0))
            })
    return filas


def graficar_metricas_evaluacion(metricas, dataset_name):
    """
    Genera gráfico de métricas globales de evaluación.
    """
    etiquetas = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    valores = [
        metricas.get('accuracy', 0.0),
        metricas.get('precision', 0.0),
        metricas.get('recall', 0.0),
        metricas.get('f1_score', 0.0)
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(etiquetas, valores, color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Valor')
    ax.set_title('Métricas de Evaluación Globales', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    for barra, valor in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2, valor + 0.02, f'{valor:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    base = _nombre_base_modelo(dataset_name)
    reporte_dir = _get_reporte_dir(dataset_name)
    ruta_grafico = reporte_dir / f"{base}_metricas_globales.png"
    plt.savefig(ruta_grafico, dpi=600, bbox_inches='tight', format='png')
    plt.close()
    print(f"✓ Gráfico de métricas globales guardado: {ruta_grafico}")
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
    filas = _obtener_metricas_por_clase(metricas)
    if not filas:
        print("⚠ No hay métricas por clase disponibles en classification_report_dict")
        return None

    clases = [f['clase'] for f in filas]
    precision = [f['precision'] for f in filas]
    recall = [f['recall'] for f in filas]
    f1 = [f['f1'] for f in filas]
    soporte = [f['support'] for f in filas]

    x = np.arange(len(clases))
    ancho = 0.25
    
    # Resolucion y calidad
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600

    # 1. Gráfico de Precisión, Recall y F1
    figsize_metrics = (14, 6) if len(clases) <= 10 else (16, 7)
    fig_metrics, ax_metrics = plt.subplots(figsize=figsize_metrics)
    
    ax_metrics.bar(x - ancho, precision, width=ancho, label='Precisión', color='#2ca02c')
    ax_metrics.bar(x, recall, width=ancho, label='Recall', color='#1f77b4')
    ax_metrics.bar(x + ancho, f1, width=ancho, label='F1-score', color='#d62728')
    ax_metrics.set_ylim(0, 1.05)
    ax_metrics.set_ylabel('Score')
    ax_metrics.set_title('Métricas de Desempeño por Clase (Precisión, Recall y F1)', fontsize=14, fontweight='bold', pad=15)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(clases, rotation=45, ha='right', fontsize=9 if len(clases) > 10 else 11)
    ax_metrics.grid(True, axis='y', alpha=0.3)
    ax_metrics.legend(loc='lower right')
    
    fig_metrics.tight_layout()
    reporte_dir = _get_reporte_dir(dataset_name)
    base = _nombre_base_modelo(dataset_name)
    ruta_metricas = reporte_dir / f"{base}_metricas_por_clase.png"
    fig_metrics.savefig(ruta_metricas, dpi=600, bbox_inches='tight', format='png')
    plt.close(fig_metrics)
    
    # 2. Gráfico de Soporte
    figsize_support = (14, 4) if len(clases) <= 10 else (16, 5)
    fig_support, ax_support = plt.subplots(figsize=figsize_support)
    
    ax_support.bar(x, soporte, color='#6c757d')
    ax_support.set_title('Soporte (Cantidad Real de Muestras de Evaluación por Clase)', fontsize=14, fontweight='bold', pad=15)
    ax_support.set_ylabel('Cantidad de Muestras')
    ax_support.set_xticks(x)
    ax_support.set_xticklabels(clases, rotation=45, ha='right', fontsize=9 if len(clases) > 10 else 11)
    ax_support.grid(True, axis='y', alpha=0.3)
    
    fig_support.tight_layout()
    ruta_soporte = reporte_dir / f"{base}_soporte_por_clase.png"
    fig_support.savefig(ruta_soporte, dpi=600, bbox_inches='tight', format='png')
    plt.close(fig_support)

    print("✓ Gráficos de métricas y soporte guardados individualmente.")
    return ruta_metricas, ruta_soporte


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
    base = _nombre_base_modelo(dataset_name)
    reporte_dir = _get_reporte_dir(dataset_name)
    ruta_grafico = reporte_dir / f"{base}_historico_completo.png"
    plt.savefig(ruta_grafico, dpi=600, bbox_inches='tight', format='png')
    plt.close()

    print(f"✓ Gráfico de histórico completo guardado: {ruta_grafico}")
    return ruta_grafico


def crear_tabla_reporte_clasificacion(metricas):
    """
    Crea una tabla estructurada con precisión, recall, F1 y soporte por clase.
    """
    reporte_dict = metricas.get('classification_report_dict', {})
    clases = metricas.get('clases', [])

    encabezado = ['Clase', 'Precisión', 'Recall', 'F1-score', 'Soporte']
    datos = [encabezado]

    if not reporte_dict:
        datos.append(['No disponible', '-', '-', '-', '-'])
    else:
        for clase in clases:
            fila = reporte_dict.get(clase)
            if not isinstance(fila, dict):
                continue
            datos.append([
                str(clase),
                f"{fila.get('precision', 0.0):.3f}",
                f"{fila.get('recall', 0.0):.3f}",
                f"{fila.get('f1-score', 0.0):.3f}",
                str(int(fila.get('support', 0)))
            ])

        for nombre in ('macro avg', 'weighted avg'):
            fila = reporte_dict.get(nombre)
            if not isinstance(fila, dict):
                continue
            datos.append([
                nombre,
                f"{fila.get('precision', 0.0):.3f}",
                f"{fila.get('recall', 0.0):.3f}",
                f"{fila.get('f1-score', 0.0):.3f}",
                str(int(fila.get('support', 0)))
            ])

    tabla = Table(datos, colWidths=[2.6*inch, 0.95*inch, 0.95*inch, 0.95*inch, 1.05*inch], repeatRows=1)
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5a9e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f7f7')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#999999')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    accuracy = reporte_dict.get('accuracy')
    texto_accuracy = f"Accuracy global: {accuracy:.3f}" if isinstance(accuracy, (int, float)) else None
    return tabla, texto_accuracy


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


def crear_tabla_detalle_clases(metricas):
    """
    Genera una tabla con las etiquetas originales agrupadas (solo two_classes).
    """
    detalle = metricas.get('clases_detalle')
    if not detalle:
        return None

    filas = [['Clase agregada', 'IDs originales', 'Etiquetas originales']]
    if isinstance(detalle, dict):
        items = detalle.items()
        for clave, etiquetas in items:
            etiquetas_txt = ", ".join(etiquetas) if isinstance(etiquetas, (list, tuple)) else str(etiquetas)
            filas.append([str(clave), "-", etiquetas_txt])
    elif isinstance(detalle, list):
        for fila in detalle:
            clave = fila.get('clase', 'N/D')
            ids = fila.get('ids_originales') or fila.get('ids') or []
            etiquetas = fila.get('etiquetas_originales') or fila.get('etiquetas') or []

            def _formatear_lista(valores):
                if not valores:
                    return "-"
                if isinstance(valores, (list, tuple)):
                    return ", ".join(str(v) for v in valores)
                return str(valores)

            filas.append([
                str(clave),
                _formatear_lista(ids),
                _formatear_lista(etiquetas)
            ])
    else:
        return None

    tabla = Table(filas, colWidths=[2.5*inch, 1.1*inch, 3.2*inch])
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fb')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#9aa5b1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return tabla


def crear_tabla_info_tecnica(metricas, dataset_name):
    """
    Crea una tabla técnica basada en artefactos reales del entrenamiento.
    """
    historico = metricas.get('historico', {})
    loss_hist = historico.get('loss', [])
    acc_hist = historico.get('accuracy', [])
    val_loss_hist = historico.get('val_loss', [])
    val_acc_hist = historico.get('val_accuracy', [])

    epocas_ejecutadas = len(loss_hist)
    mejor_val_acc = max(val_acc_hist) if val_acc_hist else None
    mejor_val_loss = min(val_loss_hist) if val_loss_hist else None
    ultima_acc_train = acc_hist[-1] if acc_hist else None
    ultima_loss_train = loss_hist[-1] if loss_hist else None
    ultima_acc_val = val_acc_hist[-1] if val_acc_hist else None
    ultima_loss_val = val_loss_hist[-1] if val_loss_hist else None

    def _fmt(v, dec=4):
        return f"{v:.{dec}f}" if isinstance(v, (int, float)) else "N/D"

    datos = [
        ['Campo técnico', 'Valor'],
        ['Dataset evaluado', dataset_name],
        ['Muestras de prueba', str(metricas.get('num_muestras_test', 'N/D'))],
        ['Clases evaluadas', str(len(metricas.get('clases', [])))],
        ['Épocas ejecutadas', str(epocas_ejecutadas if epocas_ejecutadas else 'N/D')],
        ['Accuracy test', _fmt(metricas.get('accuracy'))],
        ['Precision weighted (test)', _fmt(metricas.get('precision'))],
        ['Recall weighted (test)', _fmt(metricas.get('recall'))],
        ['F1-score weighted (test)', _fmt(metricas.get('f1_score'))],
        ['Mejor val_accuracy (histórico)', _fmt(mejor_val_acc)],
        ['Mejor val_loss (histórico)', _fmt(mejor_val_loss)],
        ['Último accuracy train', _fmt(ultima_acc_train)],
        ['Último loss train', _fmt(ultima_loss_train)],
        ['Último accuracy val', _fmt(ultima_acc_val)],
        ['Último loss val', _fmt(ultima_loss_val)],
    ]

    tabla = Table(datos, colWidths=[3.5*inch, 2.0*inch], repeatRows=1)
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#355c7d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f4f6f8')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#9aa5b1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return tabla


def crear_tabla_configuracion_entrenamiento(metricas):
    """
    Crea tabla de configuración usada durante el entrenamiento.
    """
    cfg = metricas.get('config_entrenamiento', {})
    modelo_info = metricas.get('modelo_info', {})

    datos = [
        ['Configuración', 'Valor'],
        ['Dataset', str(cfg.get('dataset_name', 'N/D'))],
        ['Batch size', str(cfg.get('batch_size', 'N/D'))],
        ['Épocas máximas', str(cfg.get('epochs_max', 'N/D'))],
        ['Test size', str(cfg.get('test_size', 'N/D'))],
        ['Validation size', str(cfg.get('validation_size', 'N/D'))],
        ['Random state', str(cfg.get('random_state', 'N/D'))],
        ['Optimizador', str(modelo_info.get('optimizador', 'N/D'))],
        ['Learning rate (inicial)', str(cfg.get('learning_rate_inicial', 'N/D'))],
        ['Learning rate (compile)', str(modelo_info.get('learning_rate', 'N/D'))],
        ['Función de pérdida', str(modelo_info.get('loss', 'N/D'))],
        ['Métricas compile', ", ".join(modelo_info.get('metricas_compile', [])) or 'N/D'],
    ]

    tabla = Table(datos, colWidths=[3.0*inch, 2.5*inch], repeatRows=1)
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#264653')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eef3f7')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#aab7c4')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return tabla


def crear_tabla_arquitectura_modelo(metricas):
    """
    Crea una tabla detallada por capa del modelo entrenado.
    """
    modelo_info = metricas.get('modelo_info', {})
    capas = modelo_info.get('capas', [])

    datos = [['#', 'Nombre', 'Tipo', 'Salida', 'Unidades', 'Activación', 'Parámetros', 'Entrenable']]
    if not capas:
        datos.append(['-', 'N/D', 'N/D', 'N/D', 'N/D', 'N/D', 'N/D', 'N/D'])
    else:
        for capa in capas:
            datos.append([
                str(capa.get('indice', '')),
                str(capa.get('nombre', '')),
                str(capa.get('tipo', '')),
                str(capa.get('salida', '')),
                str(capa.get('unidades', '')),
                str(capa.get('activacion', '')),
                str(capa.get('parametros', '')),
                'Sí' if capa.get('entrenable', False) else 'No',
            ])

    tabla = Table(
        datos,
        colWidths=[0.35*inch, 1.1*inch, 0.95*inch, 1.6*inch, 0.7*inch, 0.9*inch, 0.85*inch, 0.7*inch],
        repeatRows=1
    )
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1d3557')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (4, 1), (7, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f6fa')]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#9aa5b1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return tabla


def extraer_resumen_metricas(metricas):
    """
    Extrae un resumen de las mejores y peores clases según F1-score y las formatea
    para insertar en el PDF como texto enriquecido.
    
    Args:
        metricas (dict): Diccionario con métricas que incluye classification_report_dict
        
    Returns:
        str: HTML formateado para reportlab Paragraph
    """
    reporte_dict = metricas.get('classification_report_dict', {})
    clases = metricas.get('clases', [])
    
    # Extraer F1-scores por clase
    f1_scores = []
    for clase in clases:
        fila = reporte_dict.get(clase, {})
        if isinstance(fila, dict) and 'f1-score' in fila:
            f1_scores.append({
                'clase': clase,
                'f1': float(fila.get('f1-score', 0.0)),
                'precision': float(fila.get('precision', 0.0)),
                'recall': float(fila.get('recall', 0.0))
            })
    
    if not f1_scores:
        return "<i>No hay datos de F1-score disponibles</i>"
    
    # Ordenar por F1-score (descendente)
    f1_scores_sorted = sorted(f1_scores, key=lambda x: x['f1'], reverse=True)
    
    # Obtener top 3 y bottom 3
    top_3 = f1_scores_sorted[:3]
    bottom_3 = f1_scores_sorted[-3:] if len(f1_scores_sorted) >= 3 else f1_scores_sorted
    
    # Accuracy global
    accuracy = metricas.get('accuracy', 0.0)
    
    # Construir HTML para reportlab
    html_text = f"""
    <b>Análisis de Desempeño por Clase</b><br/>
    <br/>
    <b>Precisión Global:</b> {accuracy:.1%}<br/>
    <br/>
    <b style="color:#2ca02c">Clases con Mejor Desempeño (Top 3):</b><br/>
    """
    
    for i, item in enumerate(top_3, 1):
        html_text += f"<font color='#2ca02c'>• {item['clase']}: F1={item['f1']:.3f} (Precisión={item['precision']:.3f}, Recall={item['recall']:.3f})</font><br/>"
    
    html_text += f"""<br/>
    <b style="color:#d62728">Clases que Requieren Mejora (Bottom 3):</b><br/>
    """
    
    for i, item in enumerate(bottom_3, 1):
        html_text += f"<font color='#d62728'>• {item['clase']}: F1={item['f1']:.3f} (Precisión={item['precision']:.3f}, Recall={item['recall']:.3f})</font><br/>"
    
    return html_text


def generar_pdf(dataset_name):
    """
    Genera reporte completo en PDF.

    Args:
        dataset_name (str): Nombre del dataset
    """
    print("\n[INFO] Generando reporte PDF...")

    try:
        # Limpiar artefactos previos para evitar PDFs desactualizados
        _borrar_artefactos_previos(dataset_name)

        # Cargar datos
        metricas = cargar_metricas(dataset_name)
        y_test, y_pred, cm = cargar_predicciones(dataset_name)
        nombres_clases = [str(c) for c in metricas.get('clases', [])]
        if not nombres_clases and metricas.get('clases_detalle'):
            nombres_clases = [str(f.get('clase', '')) for f in metricas['clases_detalle']]
        tabla_detalle_clases = crear_tabla_detalle_clases(metricas)

        # Generar gráficos necesarios
        ruta_matriz_abs, ruta_matriz_norm = graficar_matriz_confusion(cm, nombres_clases, dataset_name)
        ruta_metricas_globales = graficar_metricas_evaluacion(metricas, dataset_name)
        ruta_metricas, ruta_soporte = graficar_metricas_por_clase(metricas, nombres_clases, dataset_name)
        ruta_historico = graficar_historico_completo(metricas, dataset_name)
        tabla_reporte_clasificacion, texto_accuracy = crear_tabla_reporte_clasificacion(metricas)

        # Crear documento PDF
        reporte_dir = _get_reporte_dir(dataset_name)
        ruta_pdf = reporte_dir / _nombre_pdf(dataset_name)
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
        contenido.append(Paragraph("Red Neuronal Convolucional 1D (Conv1D) — Clasificación de Actividades", subtitulo_style))
        contenido.append(Spacer(1, 0.3*inch))

        contenido.append(Paragraph(f"Dataset: <b>{dataset_name.upper()}</b>", normal_style))
        contenido.append(Paragraph(f"Fecha: <b>{datetime.now().strftime('%d/%m/%Y %H:%M')}</b>", normal_style))
        contenido.append(Spacer(1, 0.3*inch))

        # Resumen ejecutivo
        contenido.append(Paragraph("RESUMEN EJECUTIVO", subtitulo_style))
        resumen_text = f"""
        Se ha entrenado una <b>red neuronal convolucional 1D (Conv1D)</b> para la clasificación de
        actividades humanas usando datos de acelerómetro triaxial. La arquitectura consta de tres
        bloques Conv1D con BatchNormalization, MaxPooling1D y Global Average Pooling, seguida de
        una capa densa final. El entrenamiento usó <b>Focal Loss</b> con label smoothing y
        <b>GaussianNoise</b> como regularización. El modelo alcanzó una precisión de
        <b>{metricas['accuracy']:.2%}</b> en el conjunto de prueba con
        <b>{metricas['num_muestras_test']}</b> muestras evaluadas.
        Se identificaron <b>{len(metricas['clases'])}</b> clases diferentes.
        """
        contenido.append(Paragraph(resumen_text, normal_style))
        contenido.append(Spacer(1, 0.3*inch))
        if tabla_detalle_clases:
            contenido.append(Paragraph("Etiquetas incluidas en cada clase", subtitulo_style))
            contenido.append(tabla_detalle_clases)
            contenido.append(Spacer(1, 0.2*inch))

        # Tabla de métricas
        contenido.append(Paragraph("MÉTRICAS PRINCIPALES", subtitulo_style))
        contenido.append(crear_tabla_metricas(metricas))
        contenido.append(PageBreak())

        # Página 2: Gráficos globales
        contenido.append(Paragraph("MÉTRICAS DE EVALUACIÓN", subtitulo_style))
        if ruta_metricas_globales and Path(ruta_metricas_globales).exists():
            contenido.append(Image(str(ruta_metricas_globales), width=6.5*inch, height=3.3*inch))
        contenido.append(Spacer(1, 0.15*inch))

        if ruta_historico and Path(ruta_historico).exists():
            contenido.append(Paragraph("GRÁFICAS DE PRECISIÓN Y PÉRDIDA", subtitulo_style))
            contenido.append(Image(str(ruta_historico), width=6.5*inch, height=2.5*inch))
        contenido.append(PageBreak())

        # Página 3: Matriz de confusión absoluta
        contenido.append(Paragraph("MATRIZ DE CONFUSIÓN ABSOLUTA", subtitulo_style))
        if ruta_matriz_abs.exists():
            contenido.append(Image(str(ruta_matriz_abs), width=6.5*inch, height=6.0*inch))
        contenido.append(PageBreak())

        # Página 4: Matriz de confusión normalizada
        contenido.append(Paragraph("MATRIZ DE CONFUSIÓN NORMALIZADA", subtitulo_style))
        if ruta_matriz_norm.exists():
            contenido.append(Image(str(ruta_matriz_norm), width=6.5*inch, height=6.0*inch))
            contenido.append(Spacer(1, 0.15*inch))
            # Agregar resumen de métricas por clase
            resumen_html = extraer_resumen_metricas(metricas)
            contenido.append(Paragraph(resumen_html, normal_style))
        contenido.append(PageBreak())

        # Página 5: Métricas por clase
        contenido.append(Paragraph("DESEMPEÑO Y DISTRIBUCIÓN POR CLASE", subtitulo_style))
        
        explicacion_texto = """
        El siguiente gráfico detalla la <b>Precisión, Sensibilidad (Recall) y F1-Score</b> logrados para cada clase individual. 
        Estas métricas permiten comprender en qué acciones específicas el modelo tiene mayor exactitud y cuáles presentan un mayor desafío. 
        A continuación, se muestra el gráfico de <b>Soporte</b>, que indica la cantidad exacta de muestras reales que se usaron para evaluar cada clase. 
        Esto es vital para dimensionar si un desempeño bajo en una clase se debe a la falta de datos representativos o a la complejidad del movimiento.
        """
        contenido.append(Paragraph(explicacion_texto, normal_style))
        contenido.append(Spacer(1, 0.1*inch))
        
        if ruta_metricas.exists():
            contenido.append(Image(str(ruta_metricas), width=6.5*inch, height=3.0*inch))
            contenido.append(Spacer(1, 0.15*inch))
            
        if ruta_soporte.exists():
            contenido.append(Image(str(ruta_soporte), width=6.5*inch, height=2.2*inch))
            
        contenido.append(PageBreak())

        contenido.append(Paragraph("REPORTE DE CLASIFICACIÓN", subtitulo_style))
        if texto_accuracy:
            contenido.append(Paragraph(texto_accuracy, normal_style))
            contenido.append(Spacer(1, 0.08*inch))
        contenido.append(tabla_reporte_clasificacion)
        contenido.append(PageBreak())

        # Información técnica (derivada de artefactos guardados)
        contenido.append(Paragraph("INFORMACIÓN TÉCNICA", subtitulo_style))
        contenido.append(Paragraph(
            "Los siguientes campos se calculan directamente a partir de métricas e historial guardados del entrenamiento.",
            normal_style
        ))
        contenido.append(Paragraph("Configuración de entrenamiento", subtitulo_style))
        contenido.append(crear_tabla_configuracion_entrenamiento(metricas))
        contenido.append(Spacer(1, 0.15*inch))

        contenido.append(Paragraph("Resumen técnico del modelo", subtitulo_style))
        contenido.append(crear_tabla_info_tecnica(metricas, dataset_name))
        contenido.append(Spacer(1, 0.15*inch))

        modelo_info = metricas.get('modelo_info', {})
        resumen_arquitectura = (
            f"<b>Tipo:</b> Red Convolucional 1D (Conv1D) para Edge AI | "
            f"<b>Input:</b> {modelo_info.get('input_shape', 'N/D')} → Reshape (3,151) → Permute (151,3) | "
            f"<b>Output:</b> {modelo_info.get('output_shape', 'N/D')} | "
            f"<b>Bloques Conv1D:</b> 3 (64→128→256 filtros) + GlobalAvgPool + Dense(128) | "
            f"<b>Capas totales:</b> {modelo_info.get('total_capas', 'N/D')} | "
            f"<b>Parámetros:</b> {modelo_info.get('parametros_totales', 'N/D')} | "
            f"<b>Optimizador:</b> {modelo_info.get('optimizador', 'N/D')} "
            f"(lr={modelo_info.get('learning_rate', 'N/D')}) | "
            f"<b>Loss:</b> Focal Loss (alpha=0.25, gamma=2.0, label_smoothing=0.1)"
        )
        contenido.append(Paragraph("Arquitectura Conv1D por capas", subtitulo_style))
        contenido.append(Paragraph(resumen_arquitectura, normal_style))
        contenido.append(crear_tabla_arquitectura_modelo(metricas))
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

    cli_dataset = _parse_args_cli()
    env_set = "DATASET_NAME" in os.environ
    nombre_base = cli_dataset or DATASET_NAME_RAW
    dataset_name = _canonical_dataset_name(nombre_base)
    if cli_dataset:
        print(f"[INFO] Dataset por CLI: '{cli_dataset}' → '{dataset_name}'")
    elif not env_set and DATASET_NAME_RAW == "adl_fall_multiclass":
        print("[INFO] Usando dataset por defecto 'adl_fall_multiclass'. "
              "Para 9 clases ejecuta con --dataset entrenamiento_9_clases "
              "o set DATASET_NAME=entrenamiento_9_clases")
    elif dataset_name != DATASET_NAME_RAW:
        print(f"[INFO] Alias de dataset detectado: '{DATASET_NAME_RAW}' → '{dataset_name}'")

    # Generar PDF
    generar_pdf(dataset_name)


if __name__ == "__main__":
    main()
