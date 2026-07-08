# Dashboard - Proyecto Final Heart Disease
Universidad Tecnológica de Panamá | Análisis de Datos y Toma de Decisiones en Computación

## Correr localmente

```bash
pip install -r requirements.txt
python app.py
```

Luego abrir: http://localhost:8050

## Desplegar en Render (gratis)

1. Crear una cuenta en https://render.com
2. Subir esta carpeta a un repositorio de GitHub
3. En Render: **New +** → **Web Service** → conectar el repositorio
4. Configurar:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:server`
   - **Instance Type:** Free
5. Deploy. Render te da una URL pública.

## Archivos incluidos

- `app.py` — código del dashboard
- `processed_cleveland.data` — dataset Heart Disease
- `mejor_modelo_regresion.pkl` — Random Forest Regressor entrenado
- `columnas_modelo.pkl` — orden de columnas del modelo
- `scaler_regresion.pkl` — StandardScaler entrenado
- `panama_distritos.geojson` — boundaries de los 76 distritos de Panamá
- `poblacion_distritos_panama.csv` — variable sociodemográfica por distrito
- `requirements.txt` — dependencias
- `Procfile` — comando de arranque para Render/Heroku

## Contenido del dashboard

- 4 gráficas de análisis (torta, scatter, boxplot, barras)
- 2 controladores interactivos (slider de edad, radio de sexo)
- Predictor de colesterol usando Random Forest Regressor
- Mapa interactivo de distritos de Panamá con GeoPandas
- Gráfica resumen: top 10 distritos más poblados
