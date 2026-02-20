# 🗳️ Detección de Bots Políticos: Elecciones Ecuador 2025
### Redes Neuronales de Grafos (GNN) y Entropía Semántica para la Detección de Influencia Automatizada

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Descripción General

Este repositorio contiene la implementación oficial del proyecto de tesis enfocado en la detección de actores automatizados (bots) durante la campaña presidencial de Ecuador 2025 (Luisa González vs. Daniel Noboa). El dataset es original y netamente extraido mediante automatizaciones por Pedro Luzuriaga

A diferencia de los enfoques tradicionales que dependen únicamente de métricas de volumen, este framework introduce una **Arquitectura Híbrida**:
1.  **Novedad Semántica:** Utiliza la **Entropía de Intención** (Zero-Shot Classification) para medir la "rigidez" del discurso político.
2.  **Supervisión Débil:** Genera pseudo-etiquetas mediante un Índice Heurístico Multi-Vista.
3.  **Aprendizaje Topológico:** Entrena una **Graph Convolutional Network (GCN)** con Aprendizaje Sensible al Costo (Weighted Loss) para identificar bots sofisticados incrustados en el grafo social.

---

## 📂 Estructura del Repositorio

```bash
├── data.zip                   # 📦 DATASET COMPRIMIDO (Contiene:)
│   ├── Comentarios_Extraidos - Comments_Luisa.csv  # Data cruda (Luisa)
│   ├── Comentarios_Extraidos - Comments_Noboa.csv  # Data cruda (Noboa)
│   ├── Tweets archivados - Daniel_Noboa_Tweets.csv # Histórico (Noboa)
│   ├── Tweets archivados - Luisa_Gonzales_Tweets.csv # Histórico (Luisa)
│   ├── tweets_with_intents.csv                     # Data con probabilidades (BART)
│   └── tweeets.csv                                 # Dataset unificado y procesado
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb   # Limpieza e Ingeniería de Características
│   ├── 02_Intent_Analysis.ipynb      # Clasificación Zero-Shot y Cálculo de Entropía
│   └── 03_GNN_Training.ipynb         # Entrenamiento GNN y Evaluación
├── src/
│   ├── models.py              # Definición de la arquitectura GCN
│   └── utils.py               # Funciones auxiliares para métricas y ploteo
├── requirements.txt
└── README.md

```

El proyecto depende de **PyTorch** y **PyTorch Geometric**.

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/pedroluzu2001/Bot-Detection-Datasets.git](https://github.com/pedroluzu2001/Bot-Detection-Datasets.git)
    cd Bot-Detection-Datasets
    ```

2.  **⚠️ IMPORTANTE: Descomprimir Datos**
    Para que los scripts funcionen, debes descomprimir el archivo `data.zip` en la raíz del proyecto.
    * Al descomprimir, asegúrate de que quede una carpeta llamada `data/` conteniendo los archivos `.csv`.

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚡ Pipeline de Ejecución

1.  **Preprocesamiento (Opcional si usas el CSV final)**
    * `notebooks/01_Data_Preprocessing.ipynb`: Fusiona los datasets crudos de Luisa González y Daniel Noboa, limpia fechas y genera features temporales (`time_response`).

2.  **Análisis de Intención y Entropía**
    * `notebooks/02_Intent_Analysis.ipynb`: Descarga el modelo BART (`facebook/bart-large-mnli`) y calcula la **Entropía de Shannon** para cada tweet.

3.  **Etiquetado Heurístico**
    * `notebooks/03_Heuristic_Labeling.ipynb`: Genera etiquetas de entrenamiento (Weak Supervision) basándose en anomalías de entropía, antigüedad de la cuenta y tiempos de respuesta.

4.  **Entrenamiento GNN (Modelo Final)**
    * `notebooks/04_GNN_Training.ipynb`:
        * Construye el Grafo (Nodos=Usuarios, Aristas=Respuestas).
        * Calcula los pesos de clase para el balanceo.
        * Entrena la **GCN** (Hidden=64, Drop=0.6).
        * Genera las métricas de evaluación y matrices de confusión.

---

## 📊 Resultados Obtenidos

El modelo final priorizó la **Sensibilidad (Recall)** para asegurar la máxima detección de amenazas automatizadas.

| Métrica | GNN Estándar | **GNN Ponderada (Final)** | Interpretación |
| :--- | :---: | :---: | :--- |
| **Accuracy Global** | 94% | **92%** | Ligera reducción esperada por el balanceo. |
| **Precision (Bot)** | 0.87 | **0.72** | Aumento de falsos positivos (humanos tóxicos). |
| **Recall (Bot)** | 0.77 | **0.89** | **+12% en tasa de detección (Objetivo cumplido).** |

> **Hallazgo:** La auditoría cualitativa reveló que muchos "Falsos Positivos" en el modelo ponderado corresponden a usuarios humanos radicalizados que exhiben un comportamiento tóxico similar al de un bot ("Cyborgs").

---

## 📜 Cita / Referencia

Si utilizas este código o el dataset para tu investigación, por favor cita:

```bibtex
@thesis{Luzuriaga2026,
  title={Detection of Social Media Bots in Political Context using
Graph Neural Networks in Ecuador},
  year={2026},
  institution={Universidad Yachay Tech}
}
