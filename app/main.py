
## 3. Implementación del Código

### `app/main.py`

from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
import os
from datetime import datetime
import io
import uvicorn
from app.schemas.prediction import PredictionRequest, PredictionResponse, ModelInfoResponse
from app.services.predictor import SalesPredictor
from app.database import get_database
# Cargar variables de entorno
load_dotenv()
# Configuración desde .env
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
app = FastAPI(
    title="API de Predicción de Ventas",
    description="API para predecir ventas futuras basadas en datos históricos",
    version="1.0.0",
    debug=DEBUG
)

# CORS para permitir solicitudes desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las origins en desarrollo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia global del predictor
predictor = SalesPredictor()

@app.on_event("startup")
async def startup_load_model():
    """Carga el modelo al iniciar la aplicación si existe"""
    try:
        # Intentar cargar datos desde MongoDB
        predictor.load_data()
        
        # Intentar cargar modelo guardado si existe
        if os.path.exists("data/model_cache/model.joblib"):
            predictor.load_model("data/model_cache/model.joblib")
    except Exception as e:
        print(f"No se pudo cargar el modelo o los datos: {e}")
        # No es crítico, se puede entrenar un nuevo modelo con datos
@app.get("/sales-summary", tags=["Datos"])
async def get_sales_summary():
    """Devuelve un resumen de los datos de ventas disponibles"""
    
    if predictor.ventas_mensuales is None:
        # Intentar cargar datos si aún no se han cargado
        try:
            predictor.load_data()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al cargar datos: {str(e)}")
        
        if predictor.ventas_mensuales is None:
            raise HTTPException(status_code=404, detail="No se encontraron datos de ventas")
    
    # Calcular estadísticas
    total_ventas = predictor.ventas_mensuales['total'].sum()
    meses_con_datos = len(predictor.ventas_mensuales)
    promedio_mensual = total_ventas / meses_con_datos if meses_con_datos > 0 else 0
    
    # Extraer los 5 meses más recientes
    ultimos_meses = predictor.ventas_mensuales.sort_values(['año', 'mes'], ascending=False).head(5)
    ultimos_meses_datos = []
    
    for _, row in ultimos_meses.iterrows():
        ultimos_meses_datos.append({
            "año": int(row['año']),
            "mes": int(row['mes']),
            "nombre_mes": row['nombre_mes'],
            "total": float(row['total'])
        })
    
    return {
        "total_ventas": float(total_ventas),
        "meses_con_datos": meses_con_datos,
        "promedio_mensual": float(promedio_mensual),
        "ultimos_meses": ultimos_meses_datos
    }
@app.get("/", tags=["Status"])
async def root():
    """Endpoint para verificar el estado de la API"""
    return {
        "status": "online",
        "service": "API de Predicción de Ventas",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "db_connected": get_database() is not None
    }

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Modelo"])
async def get_model_info():
    """Devuelve información sobre el estado del modelo y los datos cargados"""
    if predictor.model_results is None:
        raise HTTPException(status_code=404, detail="No hay modelo entrenado")
    
    return {
        "model_name": predictor.model_results.get('mejor_nombre', 'Desconocido'),
        "r2_score": predictor.model_results.get('mejor_r2', 0),
        "rmse": predictor.model_results.get('mejor_rmse', 0),
        "data_points": len(predictor.ventas_mensuales) if predictor.ventas_mensuales is not None else 0,
        "last_trained": predictor.last_trained.strftime("%Y-%m-%d %H:%M:%S") if predictor.last_trained else None,
        "coeficientes": predictor.model_results.get('coeficientes', {}) if predictor.model_results else {}
    }

@app.post("/refresh-data", tags=["Datos"])
async def refresh_data():
    """
    Recarga los datos de ventas desde MongoDB
    """
    try:
        # Intentar cargar los datos desde MongoDB
        success = predictor.load_data()
        
        if not success:
            raise HTTPException(status_code=500, detail="Error al cargar datos desde MongoDB")
        
        return {
            "status": "success",
            "message": "Datos recargados desde MongoDB correctamente",
            "data_points": len(predictor.ventas_mensuales),
            "from_date": predictor.ventas_mensuales['fecha'].min().strftime('%Y-%m-%d') if 'fecha' in predictor.ventas_mensuales.columns else None,
            "to_date": predictor.ventas_mensuales['fecha'].max().strftime('%Y-%m-%d') if 'fecha' in predictor.ventas_mensuales.columns else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recargar datos: {str(e)}")
@app.post("/train", tags=["Modelo"])
async def train_model():
    """Entrena el modelo con los datos cargados"""
    if predictor.ventas_mensuales is None or len(predictor.ventas_mensuales) < 3:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 3 meses de datos para entrenar el modelo")
    
    try:
        # Entrenar modelo
        predictor.train_model()
        
        # Guardar modelo entrenado
        os.makedirs("data/model_cache", exist_ok=True)
        predictor.save_model("data/model_cache/model.joblib")
        
        return {
            "status": "Modelo entrenado correctamente",
            "model_info": {
                "name": predictor.model_results.get('mejor_nombre', 'Desconocido'),
                "r2_score": predictor.model_results.get('mejor_r2', 0),
                "rmse": predictor.model_results.get('mejor_rmse', 0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al entrenar el modelo: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
async def predict_sales(request: PredictionRequest):
    """
    Predice ventas para los próximos meses basado en datos históricos
    """
    if predictor.model_results is None:
        raise HTTPException(status_code=400, detail="El modelo no está entrenado. Entrene el modelo primero.")
    
    try:
        # Predecir ventas
        predicciones = predictor.predict_sales(request.meses)
        
        # Preparar resultados
        resultados = []
        
        # Agrupar predicciones por mes
        meses_unicos = predicciones['mes'].unique()
        años_unicos = predicciones['año'].unique()
        
        for mes in meses_unicos:
            for año in años_unicos:
                data_mes = predicciones[(predicciones['mes'] == mes) & (predicciones['año'] == año)]
                if len(data_mes) > 0:
                    # Obtener valores por escenario
                    pesimista = data_mes[data_mes['escenario'] == 'Pesimista']['prediccion'].iloc[0]
                    esperado = data_mes[data_mes['escenario'] == 'Esperado']['prediccion'].iloc[0]
                    optimista = data_mes[data_mes['escenario'] == 'Optimista']['prediccion'].iloc[0]
                    
                    # Añadir a resultados
                    resultados.append({
                        "mes": int(mes),
                        "año": int(año),
                        "nombre_mes": data_mes['nombre_mes'].iloc[0],
                        "prediccion_pesimista": float(pesimista),
                        "prediccion_esperada": float(esperado),
                        "prediccion_optimista": float(optimista)
                    })
        
        return {
            "predicciones": resultados,
            "total_esperado": float(predicciones[predicciones['escenario'] == 'Esperado']['prediccion'].sum()),
            "meses_predichos": len(resultados)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar predicciones: {str(e)}")

@app.get("/historical", tags=["Datos"])
async def get_historical_data():
    """Devuelve los datos históricos de ventas cargados"""
    if predictor.ventas_mensuales is None:
        raise HTTPException(status_code=404, detail="No hay datos históricos cargados")
    
    try:
        # Convertir a formato de lista para JSON
        data = []
        for _, row in predictor.ventas_mensuales.iterrows():
            data.append({
                "mes": int(row['mes']),
                "año": int(row['año']),
                "nombre_mes": row['nombre_mes'],
                "ventas": float(row['total'])
            })
        
        return {
            "historical_data": data,
            "count": len(data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar datos históricos: {str(e)}")
# Añadir el endpoint para verificar conexión a MongoDB
@app.get("/db-status", tags=["Status"])
async def check_db_status():
    try:
        db = get_database()
        collections = db.list_collection_names()
        return {
            "status": "connected",
            "database": os.getenv("DB_NAME"),
            "collections": collections,
            "ventas_count": db.ventas.count_documents({})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de conexión a MongoDB: {str(e)}")
    

# Para ejecutar desde el archivo principal
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=DEBUG)