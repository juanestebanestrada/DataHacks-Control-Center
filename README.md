# DataHacks Control Center ⚽

Centro de control interactivo para análisis de datos de fútbol, construido con Streamlit e integrado con la librería LanusStats.

## 🚀 Características

- **Extracción de Datos**: Soporte para múltiples fuentes (FBRef, FotMob, SofaScore, 365Scores, Transfermarkt)
- **Visualizaciones Interactivas**: Gráficos dinámicos con Plotly
- **Análisis Avanzado**: Simulaciones Poisson para predicción de resultados
- **Procesamiento GPU**: Aceleración con CUDA para cálculos estadísticos pesados

## 📋 Requisitos

- Python 3.8+
- Chrome/Chromium (para web scraping)
- GPU NVIDIA con CUDA (opcional, para procesamiento acelerado)

## 🔧 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/juanestebanestrada/DataHacks-Control-Center.git
cd DataHacks-Control-Center

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## ▶️ Uso

```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
├── app.py                 # Aplicación principal Streamlit
├── requirements.txt       # Dependencias del proyecto
├── utils/
│   ├── data_sources.py    # Funciones de extracción de datos
│   ├── gpu_processor.py   # Procesamiento con GPU/CUDA
│   ├── poisson_simulator.py # Simulaciones estadísticas
│   ├── sofascore_scraper.py # Web scraping de SofaScore
│   └── statsbomb_utils.py # Utilidades para datos StatsBomb
```

## 📊 Fuentes de Datos Soportadas

| Fuente | Estado |
|--------|--------|
| FBRef | ✅ Funcional |
| FotMob | ✅ Funcional |
| SofaScore | ✅ Funcional |
| 365Scores | ✅ Funcional |
| Transfermarkt | ✅ Funcional |

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios propuestos.

## 📄 Licencia

Este proyecto está bajo la licencia MIT.

---

Desarrollado con ❤️ por [Esteban](https://github.com/juanestebanestrada)
