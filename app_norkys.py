import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Configuración inicial de Streamlit
st.set_page_config(page_title="Simulador de Pronóstico de Ventas", layout="wide")
st.title("🔮 Simulador de Pronóstico de Ventas 🔮")
st.markdown("""
Bienvenido al simulador de forecasting. Esta herramienta permite pronosticar ventas en soles, evaluar modelos y tomar decisiones estratégicas.  
""")

# Verificar si Prophet está instalado
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False
    st.warning("El módulo Prophet no está instalado. Instálalo con `pip install prophet` para usar este modelo.")

# Función para calcular KPIs
def calcular_kpis(real, pronostico):
    common_dates = real.index.intersection(pronostico.index)
    real = real[common_dates]
    pronostico = pronostico[common_dates]
    mae = mean_absolute_error(real, pronostico)
    mse = mean_squared_error(real, pronostico)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((real - pronostico) / real)) * 100
    # Precisión de dirección
    real_diff = real.diff().dropna()
    pronostico_diff = pronostico.diff().dropna()
    same_direction = (real_diff > 0) == (pronostico_diff > 0)
    direccion = same_direction.mean() * 100
    # Sesgo
    sesgo = np.mean(pronostico - real)
    return {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "Precisión Dirección (%)": round(direccion, 2),
        "Sesgo": round(sesgo, 2)
    }

# Carga de datos
st.subheader("📂 Carga de Datos")
uploaded_file = st.file_uploader("Suba su dataset de ventas (Excel)", type=["xlsx"])

if uploaded_file:
    try:
        data = pd.read_excel(uploaded_file, parse_dates=['Fecha'])
        data['Fecha'] = pd.to_datetime(data['Fecha'])
        data.sort_values('Fecha', inplace=True)
        data.fillna(method='ffill', inplace=True)

        # Validación de columnas esenciales
        columnas_requeridas = ['Fecha', 'Departamento', 'Distrito', 'Canal_Venta', 'Ventas', 'Festivo', 'Eventos']
        for col in columnas_requeridas:
            if col not in data.columns:
                st.error(f"El dataset debe contener la columna '{col}'.")
                st.stop()

        # Mostrar filtros adicionales
        st.subheader("📊 Filtros")
        col1, col2, col3 = st.columns(3)
        with col1:
            departamento_options = ["Todos"] + list(data['Departamento'].unique())
            selected_depto = st.selectbox("Departamento:", departamento_options)
        with col2:
            distrito_options = ["Todos"]
            if selected_depto != "Todos":
                distrito_options += list(data[data['Departamento'] == selected_depto]['Distrito'].unique())
            selected_distrito = st.selectbox("Distrito:", distrito_options)
        with col3:
            canal_options = ["Todos"] + list(data['Canal_Venta'].unique())
            selected_canal = st.selectbox("Canal de Venta:", canal_options)

        # Aplicar filtros
        filtered_data = data.copy()
        if selected_depto != "Todos":
            filtered_data = filtered_data[filtered_data['Departamento'] == selected_depto]
        if selected_distrito != "Todos":
            filtered_data = filtered_data[filtered_data['Distrito'] == selected_distrito]
        if selected_canal != "Todos":
            filtered_data = filtered_data[filtered_data['Canal_Venta'] == selected_canal]

        # Procesamiento mensual
        data_mensual = filtered_data.resample('MS', on='Fecha').agg({
            'Ventas': 'sum',
            'Festivo': 'max'
        }).reset_index()

        # Definir TICKET_PROMEDIO
        if 'Precio_Promedio' in data.columns:
            TICKET_PROMEDIO = data['Precio_Promedio'].mean().round(2)
        else:
            TICKET_PROMEDIO = 32.47  # Valor promedio por producto

        data_mensual['Unidades'] = (data_mensual['Ventas'] / TICKET_PROMEDIO).round()
        ts_data = data_mensual.set_index('Fecha')[['Ventas', 'Unidades']]

        # Visualización histórica
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(ts_data, y='Ventas', title='Ventas en Soles', labels={'value': 'Soles', 'Fecha': ''})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(ts_data, y='Unidades', title='Unidades Vendidas', labels={'value': 'Unidades', 'Fecha': ''})
            st.plotly_chart(fig2, use_container_width=True)

        # Configuración de modelos
        st.subheader("🔮 Configuración del Pronóstico")
        col1, col2 = st.columns(2)
        with col1:
            model_options = ["Holt-Winters", "ARIMA", "SARIMA"]
            if prophet_available:
                model_options.append("Prophet")
            model_type = st.selectbox("Modelo:", model_options)
        with col2:
            confianza = st.slider("Nivel de Confianza (%):", 90, 99, 95)

        # División de datos
        train = ts_data.loc[:'2023-12-31']
        test = ts_data.loc['2024-01-01':]

        # Pronóstico y evaluación
        st.subheader("📊 Evaluación de Modelos (2024)")
        kpis_dict = {}
        forecast_test = None
        conf_int_test = None

        if model_type == "Holt-Winters":
            model = ExponentialSmoothing(train['Ventas'], seasonal='add', seasonal_periods=12, trend='add').fit()
            forecast_test = model.forecast(steps=len(test))
            stdev = np.std(model.resid)
            z_score = 1.96 if confianza == 95 else 2.33
            conf_int_test = pd.DataFrame({
                'Limite_Inferior': forecast_test - z_score * stdev,
                'Limite_Superior': forecast_test + z_score * stdev
            }, index=test.index)
            kpis_dict = calcular_kpis(test['Ventas'], forecast_test)

        elif model_type == "ARIMA":
            model = ARIMA(train['Ventas'], order=(1, 1, 1)).fit()
            forecast_obj = model.get_forecast(steps=len(test))
            forecast_test = forecast_obj.predicted_mean
            conf_int_test = forecast_obj.conf_int(alpha=1 - confianza/100)
            conf_int_test.columns = ['Limite_Inferior', 'Limite_Superior']
            kpis_dict = calcular_kpis(test['Ventas'], forecast_test)

        elif model_type == "SARIMA":
            model = SARIMAX(train['Ventas'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
            forecast_obj = model.get_forecast(steps=len(test))
            forecast_test = forecast_obj.predicted_mean
            conf_int_test = forecast_obj.conf_int(alpha=1 - confianza/100)
            conf_int_test.columns = ['Limite_Inferior', 'Limite_Superior']
            kpis_dict = calcular_kpis(test['Ventas'], forecast_test)

        elif model_type == "Prophet" and prophet_available:
            df_prophet = train.reset_index().rename(columns={'Fecha': 'ds', 'Ventas': 'y'})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(df_prophet)
            future = pd.DataFrame({'ds': test.index})
            forecast = model.predict(future)
            forecast_test = forecast.set_index('ds')['yhat']
            conf_int_test = forecast.set_index('ds')[['yhat_lower', 'yhat_upper']]
            conf_int_test.columns = ['Limite_Inferior', 'Limite_Superior']
            kpis_dict = calcular_kpis(test['Ventas'], forecast_test)
        elif model_type == "Prophet" and not prophet_available:
            st.error("El modelo Prophet no está disponible. Por favor, instale Prophet.")
            st.stop()
        else:
            st.error("Modelo no reconocido o error en la ejecución.")
            st.stop()

        # Mostrar KPIs
        st.write("**KPIs de Evaluación (2024)**")
        st.json(kpis_dict)

        # Gráfico de evaluación
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=test.index, y=test['Ventas'], name='Real', mode='lines'))
        fig_eval.add_trace(go.Scatter(x=test.index, y=forecast_test, name='Pronóstico', mode='lines'))
        if conf_int_test is not None:
            fig_eval.add_trace(go.Scatter(x=test.index, y=conf_int_test['Limite_Superior'], name='Límite Superior', mode='lines', line=dict(dash='dash')))
            fig_eval.add_trace(go.Scatter(x=test.index, y=conf_int_test['Limite_Inferior'], name='Límite Inferior', mode='lines', line=dict(dash='dash')))
        fig_eval.update_layout(title='Pronóstico vs Real (2024)', xaxis_title='Fecha', yaxis_title='Ventas (Soles)')
        st.plotly_chart(fig_eval, use_container_width=True)

        # Pronóstico 2025-2026
        st.subheader("🔮 Pronóstico 2025-2026")
        future_dates = pd.date_range(start='2025-01-01', periods=24, freq='MS')
        forecast_df = pd.DataFrame(index=future_dates)
        
        if model_type == "Holt-Winters":
            model_full = ExponentialSmoothing(ts_data['Ventas'], seasonal='add', seasonal_periods=12, trend='add').fit()
            forecast = model_full.forecast(steps=24)
            stdev = np.std(model_full.resid)
            z_score = 1.96 if confianza == 95 else 2.33
            forecast_df['Ventas'] = forecast
            forecast_df['LI_Ventas'] = forecast - z_score * stdev
            forecast_df['LS_Ventas'] = forecast + z_score * stdev

        elif model_type == "ARIMA":
            model_full = ARIMA(ts_data['Ventas'], order=(1, 1, 1)).fit()
            forecast_obj = model_full.get_forecast(steps=24)
            forecast = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int(alpha=1 - confianza/100)
            forecast_df['Ventas'] = forecast
            forecast_df['LI_Ventas'] = conf_int.iloc[:, 0]
            forecast_df['LS_Ventas'] = conf_int.iloc[:, 1]

        elif model_type == "SARIMA":
            model_full = SARIMAX(ts_data['Ventas'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
            forecast_obj = model_full.get_forecast(steps=24)
            forecast = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int(alpha=1 - confianza/100)
            forecast_df['Ventas'] = forecast
            forecast_df['LI_Ventas'] = conf_int.iloc[:, 0]
            forecast_df['LS_Ventas'] = conf_int.iloc[:, 1]

        elif model_type == "Prophet" and prophet_available:
            df_prophet_full = ts_data.reset_index().rename(columns={'Fecha': 'ds', 'Ventas': 'y'})
            model_full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model_full.fit(df_prophet_full)
            future = pd.DataFrame({'ds': future_dates})
            forecast = model_full.predict(future)
            forecast_df['Ventas'] = forecast.set_index('ds')['yhat']
            forecast_df['LI_Ventas'] = forecast.set_index('ds')['yhat_lower']
            forecast_df['LS_Ventas'] = forecast.set_index('ds')['yhat_upper']

        # Gráficos de pronóstico
        fig_forecast = px.line(forecast_df, y=['Ventas', 'LI_Ventas', 'LS_Ventas'], 
                              title='Pronóstico de Ventas (2025-2026)', labels={'value': 'Soles', 'Fecha': ''})
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Descarga de resultados
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            forecast_df.to_excel(writer, sheet_name='Pronostico')
        buffer.seek(0)
        st.download_button(
            label="⬇️ Descargar Pronóstico (Excel)",
            data=buffer,
            file_name=f"pronostico_{model_type}_2025_2026.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Notas educativas
        st.subheader("📚 Notas para Estudiantes")
        st.markdown("""
        - **MAE**: Promedio de errores absolutos. Menor es mejor.
        - **RMSE**: Penaliza errores grandes. Útil para detectar outliers.
        - **MAPE**: Error en porcentaje. Ideal para comparar entre locales.
        - **Precisión de Dirección**: Indica si el modelo predice correctamente alzas o bajas.
        - **Sesgo**: Positivo (sobreestima), Negativo (subestima).

        
        """)
    except Exception as e:
        st.error(f"Error al cargar el dataset: {str(e)}")
else:
    st.info("Por favor, suba un archivo Excel con sus datos de ventas.")

# Footer
st.markdown("""
<div style='text-align: center; font-size: 12px; margin-top: 50px; color: #666;'>
    ©️ 2025 Diseñado por <b>Wilton Torvisco</b> | 
    Todos los derechos reservados.
</div>
""", unsafe_allow_html=True)
   
