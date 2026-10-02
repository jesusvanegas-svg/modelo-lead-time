import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Modelo Predictivo de Lead Time - Spradling",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Modelo Predictivo de Analítica Avanzada: Lead Time de Planta")
st.markdown("""
**Proyecto de Grado:** Optimización de Tiempos de Entrega mediante Machine Learning (*Random Forest Regressor*).  
**Meta Operativa:** Entregar pedidos en $\le 17$ días (Principio *Just in Time*).
""")

# ---------------------------------------------------------
# CARGA Y PREPROCESAMIENTO DE DATOS
# ---------------------------------------------------------
@st.cache_data
def cargar_y_preprocesar_datos():
    # 1. Cargar bases de datos
    df_lead = pd.read_excel('LEAD TIME DE PLANTA 2023-2026 25 SEPTIWMBRE.xlsx')
    df_paradas = pd.read_excel('PARADAS DE MAQUINA 2024-2026.....xlsx', header=1)
    
    # Limpiar espacios en blanco en los nombres de las columnas
    df_lead.columns = df_lead.columns.str.strip()
    df_paradas.columns = df_paradas.columns.str.strip()
    
    # Depurar filas sin llave o sin variable objetivo
    df_lead = df_lead.dropna(subset=['Orden', 'Tiempo total en dias', 'Metros a Fabri', 'Maquina trabaajo', 'Linea /Acabados'])
    df_paradas = df_paradas.dropna(subset=['Orden', 'Minutos'])
    
    # 2. Consolidador de paradas por orden
    paradas_agrupadas = df_paradas.groupby('Orden')['Minutos'].sum().reset_index()
    paradas_agrupadas.rename(columns={'Minutos': 'Total_Minutos_Parada'}, inplace=True)
    
    # 3. Mezclar fuentes de datos
    df_merged = pd.merge(df_lead, paradas_agrupadas, on='Orden', how='left')
    df_merged['Total_Minutos_Parada'] = df_merged['Total_Minutos_Parada'].fillna(0)
    
    # 4. Ingenieria de Características (Feature Engineering Temporal)
    df_merged['Fecha_DT'] = pd.to_datetime(df_merged['Fecha.Geneneración.'], errors='coerce')
    df_merged['Mes_Generacion'] = df_merged['Fecha_DT'].dt.month.fillna(0).astype(int).astype(str)
    df_merged['Dia_Semana'] = df_merged['Fecha_DT'].dt.day_name().fillna('DESCONOCIDO')
    
    # 5. Seleccionar variables predictoras y objetivo
    features = ['Metros a Fabri', 'Maquina trabaajo', 'Linea /Acabados', 'Reprocesada', 'Total_Minutos_Parada', 'Mes_Generacion', 'Dia_Semana']
    X = df_merged[features].copy()
    y = df_merged['Tiempo total en dias']
    
    # Estructurar variables categóricas
    for col in ['Maquina trabaajo', 'Linea /Acabados', 'Reprocesada', 'Mes_Generacion', 'Dia_Semana']:
        X[col] = X[col].astype(str).str.strip().str.upper()
        
    return X, y, df_merged

X, y, df_merged = cargar_y_preprocesar_datos()

# ---------------------------------------------------------
# CONSTRUCCIÓN DEL MODELO DE MACHINE LEARNING
# ---------------------------------------------------------
categorical_features = ['Maquina trabaajo', 'Linea /Acabados', 'Reprocesada', 'Mes_Generacion', 'Dia_Semana']
numeric_features = ['Metros a Fabri', 'Total_Minutos_Parada']

# Preprocesador con Escalado Estándar y One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Pipeline de Ensamble con Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
])

# Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------------------------------------------
# PANEL LATERAL: SIMULADOR INTERACTIVO DE ÓRDENES
# ---------------------------------------------------------
st.sidebar.header("🕹️ Simular Nueva Orden de Producción")

metros_in = st.sidebar.number_input("Metros a Fabricar", min_value=50, max_value=100000, value=5000, step=500)
maquina_in = st.sidebar.selectbox("Máquina Asignada", sorted(X['Maquina trabaajo'].unique()))
linea_in = st.sidebar.selectbox("Línea de Acabado", sorted(X['Linea /Acabados'].unique()))
reproceso_in = st.sidebar.selectbox("¿Es Reproceso?", ["NO", "SI"])
parada_in = st.sidebar.number_input("Minutos de Parada Estimados", min_value=0, max_value=5000, value=60, step=30)
mes_in = st.sidebar.selectbox("Mes de Generación", sorted(X['Mes_Generacion'].unique()))
dia_in = st.sidebar.selectbox("Día de la Semana", sorted(X['Dia_Semana'].unique()))

btn_predict = st.sidebar.button("🚀 Predecir Lead Time", use_container_width=True)

# ---------------------------------------------------------
# PRESENTACIÓN DE RESULTADOS DE SIMULACIÓN
# ---------------------------------------------------------
if btn_predict:
    input_df = pd.DataFrame({
        'Metros a Fabri': [metros_in],
        'Maquina trabaajo': [str(maquina_in)],
        'Linea /Acabados': [str(linea_in)],
        'Reprocesada': [str(reproceso_in)],
        'Total_Minutos_Parada': [parada_in],
        'Mes_Generacion': [str(mes_in)],
        'Dia_Semana': [str(dia_in)]
    })
    
    prediccion_dias = model.predict(input_df)[0]
    
    st.subheader("📌 Resultado de la Predicción para la Orden Simulado")
    col_res1, col_res2, col_res3 = st.columns(3)
    
    col_res1.metric("Lead Time Proyectado", f"{prediccion_dias:.1f} Días")
    
    diferencia_meta = prediccion_dias - 17
    if prediccion_dias > 17:
        col_res2.metric("Estado de Cumplimiento", "⚠️ RETRASO", delta=f"+{diferencia_meta:.1f} días sobre la meta", delta_color="inverse")
        col_res3.error("El pedido supera los 17 días máximos. Se sugiere revisar reprogramación o reducir paradas.")
    else:
        col_res2.metric("Estado de Cumplimiento", "✅ A TIEMPO", delta=f"{diferencia_meta:.1f} días respecto a la meta", delta_color="normal")
        col_res3.success("El pedido cumple rigurosamente con los objetivos Just in Time (JIT).")

st.markdown("---")

# ---------------------------------------------------------
# SECCIÓN PARA EL JURADO: IMPORTANCIA DE VARIABLES
# ---------------------------------------------------------
st.subheader("📈 Análisis de Importancia de Variables (Feature Importance)")
st.markdown("""
Este gráfico ilustra cuantitativamente el peso porcentual de cada factor productivo en las decisiones del modelo. 
Permite identificar directamente qué variables provocan los mayores cuellos de botella en la planta.
""")

# Extraer nombres de las variables codificadas
cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
cat_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_names)
importances = model.named_steps['regressor'].feature_importances_ * 100

df_imp = pd.DataFrame({'Variable_Codificada': all_feature_names, 'Importancia': importances})

# Mapeo a categorías agrupadas para lectura clara
def mapear_categoria(var):
    if var in ['Metros a Fabri', 'Total_Minutos_Parada']:
        return var
    elif 'Reprocesada' in var:
        return 'Condición de Reproceso'
    elif 'Maquina' in var:
        return 'Máquina de Trabajo'
    elif 'Linea' in var:
        return 'Línea de Acabado'
    elif 'Mes' in var:
        return 'Estacionalidad (Mes)'
    elif 'Dia' in var:
        return 'Día de la Semana'
    return 'Otros'

df_imp['Grupo_Factor'] = df_imp['Variable_Codificada'].apply(mapear_categoria)
df_grouped_imp = df_imp.groupby('Grupo_Factor')['Importancia'].sum().reset_index()
df_grouped_imp = df_grouped_imp.sort_values(by='Importancia', ascending=True)

fig_imp = px.bar(
    df_grouped_imp,
    x='Importancia',
    y='Grupo_Factor',
    orientation='h',
    text=df_grouped_imp['Importancia'].apply(lambda x: f"{x:.2f}%"),
    title="<b>Impacto Porcentual Relativo por Factor Operativo</b>",
    color='Importancia',
    color_continuous_scale='Reds'
)
fig_imp.update_layout(xaxis_title="Impacto en el Lead Time (%)", yaxis_title="Factor de Planta", showlegend=False, height=400)
fig_imp.update_traces(textposition='outside')

st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# EVALUACIÓN MATEMÁTICA Y MÉTRICAS DE VALIDACIÓN
# ---------------------------------------------------------
st.subheader("🧪 Métricas Matemáticas de Evaluación del Modelo")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

col_m1.metric("Error Absoluto Medio (MAE)", f"{mae:.2f} días", help="Promedio de error directo en días.")
col_m2.metric("Raíz Error Cuadrático (RMSE)", f"{rmse:.2f} días", help="Penaliza desviaciones o picos atípicos.")
col_m3.metric("Coeficiente $R^2$", f"{r2:.2f}", help="Proporción de variabilidad explicada por el modelo.")
col_m4.metric("Registros Analizados", f"{len(df_merged):,}", help="Total de órdenes de producción cruzadas.")