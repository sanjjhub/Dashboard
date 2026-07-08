"""
Dashboard - Proyecto Final
Análisis de Datos y Toma de Decisiones en Computación
Universidad Tecnológica de Panamá

Dataset: Heart Disease Cleveland (UCI)
Autor: Miguel (E-8-212418)
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COLUMNAS = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
            'exang','oldpeak','slope','ca','thal','target']

df = pd.read_csv(os.path.join(BASE_DIR, 'processed_cleveland.data'),
                 header=None, names=COLUMNAS, na_values='?')

imputer = SimpleImputer(strategy='median')
df = pd.DataFrame(imputer.fit_transform(df), columns=COLUMNAS)
df['target_bin'] = (df['target'] > 0).astype(int)
df['diagnostico'] = df['target_bin'].map({0: 'Sin Enfermedad', 1: 'Con Enfermedad'})

with open(os.path.join(BASE_DIR, 'mejor_modelo_regresion.pkl'), 'rb') as f:
    modelo_reg = pickle.load(f)
with open(os.path.join(BASE_DIR, 'columnas_modelo.pkl'), 'rb') as f:
    columnas_modelo = pickle.load(f)

# Mapa: leer GeoJSON directo sin geopandas
with open(os.path.join(BASE_DIR, 'panama_distritos.geojson'), encoding='utf-8') as f:
    geojson_data = json.load(f)

# Agregar id a cada feature para que plotly los pueda referenciar
for feat in geojson_data['features']:
    feat['id'] = feat['properties']['shapeName']

poblacion = pd.read_csv(os.path.join(BASE_DIR, 'poblacion_distritos_panama.csv'))

# ---------------------------------------------------------------------------
# 2. APP DASH
# ---------------------------------------------------------------------------

app = Dash(__name__)
server = app.server
app.title = 'Dashboard Heart Disease - UTP'

COLOR_BG = '#f5f7fa'
COLOR_CARD = 'white'
COLOR_PRIMARIO = '#2c3e50'
COLOR_SEC = '#3498db'

card_style = {
    'backgroundColor': COLOR_CARD,
    'padding': '20px',
    'borderRadius': '8px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
    'marginBottom': '20px'
}

# ---------------------------------------------------------------------------
# 3. LAYOUT
# ---------------------------------------------------------------------------

app.layout = html.Div(style={'backgroundColor': COLOR_BG, 'padding': '20px',
                             'fontFamily': 'Arial, sans-serif'}, children=[

    # Encabezado
    html.Div(style={**card_style, 'textAlign': 'center', 'backgroundColor': COLOR_PRIMARIO,
                    'color': 'white'}, children=[
        html.H1('Dashboard - Análisis de Enfermedad Cardíaca',
                style={'margin': '0', 'fontSize': '28px'}),
        html.P('Universidad Tecnológica de Panamá | Proyecto Final',
               style={'margin': '5px 0 0 0', 'fontSize': '14px'}),
        html.P('¿Es posible predecir enfermedad cardíaca a partir de indicadores clínicos?',
               style={'margin': '5px 0 0 0', 'fontSize': '13px', 'fontStyle': 'italic'})
    ]),

    # Controladores
    html.Div(style=card_style, children=[
        html.H3('Filtros Interactivos', style={'color': COLOR_PRIMARIO}),
        html.Div(style={'display': 'flex', 'gap': '40px', 'alignItems': 'center'}, children=[
            html.Div(style={'flex': '2'}, children=[
                html.Label('Rango de Edad:', style={'fontWeight': 'bold'}),
                dcc.RangeSlider(
                    id='slider-edad',
                    min=int(df['age'].min()), max=int(df['age'].max()), step=1,
                    value=[int(df['age'].min()), int(df['age'].max())],
                    marks={i: str(i) for i in range(30, 80, 10)},
                    tooltip={'placement': 'bottom', 'always_visible': False}
                )
            ]),
            html.Div(style={'flex': '1'}, children=[
                html.Label('Sexo:', style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='radio-sexo',
                    options=[
                        {'label': ' Todos', 'value': 'todos'},
                        {'label': ' Masculino', 'value': 'M'},
                        {'label': ' Femenino', 'value': 'F'}
                    ],
                    value='todos', inline=True,
                    labelStyle={'marginRight': '15px'}
                )
            ])
        ])
    ]),

    # 4 gráficas de análisis
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
        html.Div(style=card_style, children=[
            html.H4('1. Distribución del Diagnóstico', style={'color': COLOR_PRIMARIO}),
            dcc.Graph(id='grafica-diagnostico')
        ]),
        html.Div(style=card_style, children=[
            html.H4('2. Edad vs Frecuencia Cardíaca Máxima', style={'color': COLOR_PRIMARIO}),
            dcc.Graph(id='grafica-scatter')
        ]),
        html.Div(style=card_style, children=[
            html.H4('3. Colesterol por Diagnóstico', style={'color': COLOR_PRIMARIO}),
            dcc.Graph(id='grafica-boxplot')
        ]),
        html.Div(style=card_style, children=[
            html.H4('4. Tipo de Dolor de Pecho por Diagnóstico', style={'color': COLOR_PRIMARIO}),
            dcc.Graph(id='grafica-barras')
        ]),
    ]),

    # Predictor de colesterol
    html.Div(style=card_style, children=[
        html.H3('Predictor de Colesterol (Random Forest Regressor)',
                style={'color': COLOR_PRIMARIO}),
        html.P('Ingrese los datos del paciente para predecir su nivel estimado de colesterol (mg/dl).',
               style={'fontSize': '13px', 'color': '#666'}),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '15px'}, children=[
            html.Div([html.Label('Edad:'), dcc.Input(id='in-age', type='number', value=54, min=20, max=100, style={'width': '100%'})]),
            html.Div([html.Label('Sexo (0=F, 1=M):'), dcc.Input(id='in-sex', type='number', value=1, min=0, max=1, style={'width': '100%'})]),
            html.Div([html.Label('Tipo dolor pecho (1-4):'), dcc.Input(id='in-cp', type='number', value=3, min=1, max=4, style={'width': '100%'})]),
            html.Div([html.Label('Presión arterial:'), dcc.Input(id='in-trestbps', type='number', value=130, min=80, max=220, style={'width': '100%'})]),
            html.Div([html.Label('Azúcar >120 (0/1):'), dcc.Input(id='in-fbs', type='number', value=0, min=0, max=1, style={'width': '100%'})]),
            html.Div([html.Label('ECG reposo (0-2):'), dcc.Input(id='in-restecg', type='number', value=1, min=0, max=2, style={'width': '100%'})]),
            html.Div([html.Label('Frec. cardíaca máx:'), dcc.Input(id='in-thalach', type='number', value=150, min=70, max=220, style={'width': '100%'})]),
            html.Div([html.Label('Angina ejercicio (0/1):'), dcc.Input(id='in-exang', type='number', value=0, min=0, max=1, style={'width': '100%'})]),
            html.Div([html.Label('Depresión ST:'), dcc.Input(id='in-oldpeak', type='number', value=1.0, step=0.1, style={'width': '100%'})]),
            html.Div([html.Label('Pendiente ST (1-3):'), dcc.Input(id='in-slope', type='number', value=2, min=1, max=3, style={'width': '100%'})]),
            html.Div([html.Label('Vasos coloreados (0-3):'), dcc.Input(id='in-ca', type='number', value=0, min=0, max=3, style={'width': '100%'})]),
            html.Div([html.Label('Talasemia (3, 6, 7):'), dcc.Input(id='in-thal', type='number', value=3, min=3, max=7, style={'width': '100%'})]),
        ]),
        html.Br(),
        html.Button('Predecir Colesterol', id='btn-predecir', n_clicks=0,
                    style={'backgroundColor': COLOR_SEC, 'color': 'white',
                           'border': 'none', 'padding': '10px 25px',
                           'borderRadius': '5px', 'cursor': 'pointer', 'fontSize': '15px'}),
        html.Br(), html.Br(),
        html.Div(id='resultado-prediccion', style={'fontSize': '20px', 'fontWeight': 'bold', 'color': COLOR_PRIMARIO})
    ]),

    # Mapa de Panamá
    html.Div(style=card_style, children=[
        html.H3('Mapa Interactivo - Población por Distrito (Panamá)', style={'color': COLOR_PRIMARIO}),
        html.P('Variable sociodemográfica: Población total por distrito. Fuente: INEC Panamá / geoBoundaries.org',
               style={'fontSize': '13px', 'color': '#666'}),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '2fr 1fr', 'gap': '20px'}, children=[
            dcc.Graph(id='mapa-panama'),
            dcc.Graph(id='top-distritos')
        ])
    ]),

    html.Div(style={'textAlign': 'center', 'padding': '15px', 'color': '#888', 'fontSize': '12px'}, children=[
        html.P('Proyecto Final 2026 - Análisis de Datos y Toma de Decisiones en Computación'),
        html.P('Facilitador: José Carlos Rangel Ortiz, PhD. | UTP')
    ])
])

# ---------------------------------------------------------------------------
# 4. CALLBACKS
# ---------------------------------------------------------------------------

def filtrar_df(rango_edad, sexo):
    d = df[(df['age'] >= rango_edad[0]) & (df['age'] <= rango_edad[1])]
    if sexo == 'M':
        d = d[d['sex'] == 1.0]
    elif sexo == 'F':
        d = d[d['sex'] == 0.0]
    return d


@app.callback(
    [Output('grafica-diagnostico', 'figure'),
     Output('grafica-scatter', 'figure'),
     Output('grafica-boxplot', 'figure'),
     Output('grafica-barras', 'figure')],
    [Input('slider-edad', 'value'), Input('radio-sexo', 'value')]
)
def actualizar_graficas(rango_edad, sexo):
    d = filtrar_df(rango_edad, sexo)

    if len(d) == 0:
        fig_v = go.Figure().add_annotation(text='Sin datos', showarrow=False)
        return fig_v, fig_v, fig_v, fig_v

    # 1. Torta diagnóstico
    conteo = d['diagnostico'].value_counts()
    fig1 = px.pie(values=conteo.values, names=conteo.index,
                  color=conteo.index,
                  color_discrete_map={'Sin Enfermedad': '#2ecc71', 'Con Enfermedad': '#e74c3c'},
                  hole=0.4)
    fig1.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300)

    # 2. Scatter edad vs thalach
    fig2 = px.scatter(d, x='age', y='thalach', color='diagnostico',
                      color_discrete_map={'Sin Enfermedad': '#2ecc71', 'Con Enfermedad': '#e74c3c'},
                      labels={'age': 'Edad', 'thalach': 'Frec. Cardíaca Máx.'}, opacity=0.7)
    fig2.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300,
                       legend=dict(orientation='h', y=-0.25))

    # 3. Boxplot colesterol
    fig3 = px.box(d, x='diagnostico', y='chol', color='diagnostico',
                  color_discrete_map={'Sin Enfermedad': '#2ecc71', 'Con Enfermedad': '#e74c3c'},
                  labels={'chol': 'Colesterol (mg/dl)', 'diagnostico': ''})
    fig3.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300, showlegend=False)

    # 4. Barras tipo dolor de pecho
    cp_labels = {1.0: 'Angina típica', 2.0: 'Angina atípica', 3.0: 'No angina', 4.0: 'Asintomático'}
    d = d.copy()
    d['cp_str'] = d['cp'].map(cp_labels)
    conteo_cp = d.groupby(['cp_str', 'diagnostico']).size().reset_index(name='cantidad')
    fig4 = px.bar(conteo_cp, x='cp_str', y='cantidad', color='diagnostico',
                  color_discrete_map={'Sin Enfermedad': '#2ecc71', 'Con Enfermedad': '#e74c3c'},
                  barmode='group', labels={'cp_str': 'Tipo de dolor', 'cantidad': 'Cantidad'})
    fig4.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300,
                       legend=dict(orientation='h', y=-0.3))

    return fig1, fig2, fig3, fig4


@app.callback(
    Output('resultado-prediccion', 'children'),
    Input('btn-predecir', 'n_clicks'),
    [Input('in-age', 'value'), Input('in-sex', 'value'), Input('in-cp', 'value'),
     Input('in-trestbps', 'value'), Input('in-fbs', 'value'), Input('in-restecg', 'value'),
     Input('in-thalach', 'value'), Input('in-exang', 'value'), Input('in-oldpeak', 'value'),
     Input('in-slope', 'value'), Input('in-ca', 'value'), Input('in-thal', 'value')]
)
def predecir_colesterol(n_clicks, age, sex, cp, trestbps, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal):
    if n_clicks == 0:
        return 'Ingrese los datos del paciente y presione "Predecir Colesterol".'
    try:
        instancia = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
            'ca': ca, 'thal': thal
        }])[columnas_modelo]
        pred = modelo_reg.predict(instancia)[0]
        return f'✓ Colesterol Predicho: {pred:.1f} mg/dl'
    except Exception as e:
        return f'Error: {str(e)}'


@app.callback(
    [Output('mapa-panama', 'figure'), Output('top-distritos', 'figure')],
    Input('slider-edad', 'value')
)
def actualizar_mapa(_):
    # Mapa coroplético usando GeoJSON directo (sin geopandas)
    fig_mapa = px.choropleth_mapbox(
        poblacion,
        geojson=geojson_data,
        locations='distrito',
        color='poblacion',
        color_continuous_scale='Viridis',
        mapbox_style='carto-positron',
        center={'lat': 8.5, 'lon': -80.5},
        zoom=6,
        opacity=0.75,
        hover_name='distrito',
        hover_data={'poblacion': ':,'},
        labels={'poblacion': 'Población'}
    )
    fig_mapa.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)

    # Top 10 distritos
    top10 = poblacion.nlargest(10, 'poblacion').sort_values('poblacion')
    fig_top = px.bar(top10, x='poblacion', y='distrito', orientation='h',
                     color='poblacion', color_continuous_scale='Viridis',
                     labels={'poblacion': 'Población', 'distrito': ''},
                     title='Top 10 Distritos Más Poblados')
    fig_top.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=500,
                          showlegend=False, coloraxis_showscale=False)

    return fig_mapa, fig_top


# ---------------------------------------------------------------------------
# 5. RUN
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
