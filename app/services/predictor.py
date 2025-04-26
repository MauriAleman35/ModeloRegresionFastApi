import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os

from app.database import get_sales_data
from app.models.prediction import SalesPredictionModel

class SalesPredictor:
    """Servicio para predicción de ventas"""
    
    def __init__(self):
        self.ventas_mensuales = None
        self.model = SalesPredictionModel()
        self.model_results = None
        self.last_trained = None
    
    def load_data(self, path_csv=None):
    
        try:
             # Siempre intentar cargar desde MongoDB primero
            print("Cargando datos desde MongoDB...")
            self.ventas_mensuales = get_sales_data()
        
            if self.ventas_mensuales is not None and len(self.ventas_mensuales) > 0:
                print(f"Datos cargados desde MongoDB: {len(self.ventas_mensuales)} registros")
                return True
            if path_csv:
                print(f"Intentando cargar desde CSV: {path_csv}")
                 # ... código para cargar desde CSV ...
                return True
            else:
                print("No se pudieron cargar datos desde MongoDB y no se proporcionó CSV")
                return False
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return False
    
    def prepare_data_model(self):
        """Prepara los datos para el modelo de predicción"""
        if self.ventas_mensuales is None:
            raise ValueError("No hay datos cargados")
        
        # Trabajar con una copia
        df = self.ventas_mensuales.copy()
        
        # Variables de tiempo básicas
        df['tiempo'] = (df['año'] - df['año'].min()) * 12 + df['mes']
        df['tiempo_sq'] = df['tiempo'] ** 2  # Componente cuadrático para tendencias no lineales
        
        # Codificación estacional específica para mercado
        df['temporada_alta'] = df['mes'].isin([2, 12]).astype(int)
        df['temporada_media'] = df['mes'].isin([3, 7, 11]).astype(int)
        df['temporada_baja'] = df['mes'].isin([4, 5, 6, 9, 10]).astype(int)
        df['temporada_muy_baja'] = df['mes'].isin([1, 8]).astype(int)
        
        # Definir características según cantidad de datos disponibles
        if len(df) >= 12:  # Si tenemos un año o más de datos
            características = ['tiempo', 'tiempo_sq', 'temporada_alta', 'temporada_media', 'temporada_baja']
        elif len(df) >= 6:  # Si tenemos medio año o más
            características = ['tiempo', 'tiempo_sq', 'temporada_alta']
        else:  # Con menos datos, modelo muy simple
            características = ['tiempo', 'temporada_alta']
        
        # Preparar matrices
        X = df[características]
        y = df['total']
        
        return X, y, características
    
    def train_model(self):
        """Entrena el modelo de predicción"""
        X, y, _ = self.prepare_data_model()
        
        # Entrenar modelo
        self.model.fit(X, y)
        
        # Guardar resultados
        self.model_results = self.model.model_info
        self.last_trained = datetime.now()
        
        print(f"Modelo entrenado con éxito: R² = {self.model_results['mejor_r2']:.4f}")
        return self.model_results
    
    # [El resto del código se mantiene igual]
    # Métodos para predicción de ventas, etc.

    def predict_sales(self, meses_a_predecir=6):
        """Genera predicciones de ventas para los próximos meses"""
        if self.ventas_mensuales is None:
            raise ValueError("No hay datos históricos")
            
        if self.model_results is None:
            raise ValueError("El modelo no está entrenado")
        
        # Obtener predicciones base
        predicciones_modelo = self._predict_base_sales(meses_a_predecir)
        
        # Preparar datos históricos
        df_temp = self.ventas_mensuales.copy().sort_values(['año', 'mes'])
        
        # Estadísticas históricas importantes
        ventas_por_mes = {}
        for _, row in df_temp.iterrows():
            ventas_por_mes[row['mes']] = row['total']
        
        ultimo_mes = df_temp.iloc[-1]
        ultimo_valor = ultimo_mes['total']
        media_ventas = df_temp['total'].mean()
        
        # Factores estacionales
        factores_estacionales = {
            2: 1.50,  # Febrero (preparación regreso a clases)
            12: 1.40, # Diciembre (navidad)
            3: 1.25,  # Marzo (inicio de clases)
            11: 1.20, # Noviembre (preparación cierre año)
            7: 1.15,  # Julio (receso invernal)
            10: 1.05, # Octubre (incluye Halloween)
            4: 0.90,  # Abril
            5: 0.85,  # Mayo
            6: 0.90,  # Junio
            9: 0.95,  # Septiembre
            1: 0.80,  # Enero (post navidad)
            8: 0.75,  # Agosto (post receso)
        }
        
        # GENERAR LOS TRES ESCENARIOS
        predicciones_combinadas = []
        
        meses = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
                 7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
        
        # Para cada mes a predecir
        for i, row in predicciones_modelo.iterrows():
            mes_nuevo = row['mes']
            año_nuevo = row['año']
            factor_estacional = factores_estacionales.get(mes_nuevo, 1.0)
            
            # [Generación de escenarios pesimista, esperado y optimista]
            # Lógica para los tres escenarios que ya tienes implementada
            # ...
            
            # Ejemplo simplificado
            prediccion_pesimista = row['prediccion'] * 0.8
            prediccion_esperado = row['prediccion'] 
            prediccion_optimista = row['prediccion'] * 1.2
            
            # Guardar los tres escenarios
            for escenario, prediccion in [
                ('Pesimista', prediccion_pesimista),
                ('Esperado', prediccion_esperado),
                ('Optimista', prediccion_optimista)
            ]:
                predicciones_combinadas.append({
                    'año': año_nuevo,
                    'mes': mes_nuevo,
                    'nombre_mes': meses[mes_nuevo],
                    'prediccion': prediccion,
                    'escenario': escenario
                })
        
        # Convertir a DataFrame
        df_predicciones = pd.DataFrame(predicciones_combinadas)
        return df_predicciones
    
    # Método auxiliar para predicción base
    def _predict_base_sales(self, meses_a_predecir):
        # Implementación específica para la predicción base
        # ...
        
        # Código simplificado de ejemplo para ilustrar
        ultimo_mes = self.ventas_mensuales.sort_values(['año', 'mes']).iloc[-1]
        
        predicciones = []
        for i in range(1, meses_a_predecir + 1):
            mes_nuevo = ((ultimo_mes['mes'] + i - 1) % 12) + 1
            año_nuevo = ultimo_mes['año'] + ((ultimo_mes['mes'] + i - 1) // 12)
            
            # Usar el modelo para predecir
            # ...
            
            # Ejemplo simple (incremento)
            prediccion = ultimo_mes['total'] * (1 + 0.05 * i)
            
            meses = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
                     7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
                
            predicciones.append({
                'año': año_nuevo,
                'mes': mes_nuevo,
                'nombre_mes': meses[mes_nuevo],
                'prediccion': prediccion
            })
            
        return pd.DataFrame(predicciones)
    
    def save_model(self, path):
        """Guarda el modelo entrenado"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Modelo guardado en {path}")
    
    def load_model(self, path):
        """Carga un modelo guardado previamente"""
        self.model.load(path)
        self.model_results = self.model.model_info
        self.last_trained = self.model.model_info.get('last_trained', datetime.now())
        print(f"Modelo cargado desde {path}")