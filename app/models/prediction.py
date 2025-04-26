import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime

class SalesPredictionModel:
    """Modelo de predicción de ventas"""
    def __init__(self):
        self.scaler = None
        self.model = None
        self.model_info = {
            'mejor_modelo': None,
            'mejor_nombre': None,
            'mejor_rmse': float('inf'),
            'mejor_r2': -float('inf'),
            'coeficientes': {},
            'last_trained': None
        }
    
    def fit(self, X, y):
        """Entrena el modelo con los datos proporcionados"""
        # Escalar características
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Para pocos datos, probar modelos con diferentes niveles de regularización
        modelos = {
            'Regresión Lineal Simple': LinearRegression(),
            'Ridge (alpha=0.1)': Ridge(alpha=0.1),
            'Ridge (alpha=1.0)': Ridge(alpha=1.0)
        }
        
        # Entrenamiento y evaluación
        for nombre, modelo in modelos.items():
            try:
                # Ajustar modelo
                modelo.fit(X_scaled, y)
                
                # Predecir sobre los mismos datos
                y_pred = modelo.predict(X_scaled)
                
                # Calcular métricas
                rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
                r2 = float(r2_score(y, y_pred))
                
                # Si r2 es NaN, asignar un valor bajo
                if np.isnan(r2):
                    r2 = -1.0
                
                print(f"Modelo {nombre}: RMSE = {rmse:.2f}, R² = {r2:.2f}")
                
                # Actualizar mejor modelo
                if r2 > self.model_info['mejor_r2']:
                    self.model_info['mejor_modelo'] = modelo
                    self.model_info['mejor_rmse'] = rmse
                    self.model_info['mejor_r2'] = r2
                    self.model_info['mejor_nombre'] = nombre
                    self.model = modelo
            except Exception as e:
                print(f"Error en modelo {nombre}: {str(e)}")
                continue
        
        # Guardar coeficientes
        if self.model is not None and hasattr(self.model, 'coef_'):
            self.model_info['coeficientes'] = {}
            for i, col_name in enumerate(X.columns):
                self.model_info['coeficientes'][col_name] = float(self.model.coef_[i])
        
        self.model_info['last_trained'] = datetime.now()
        
        return self
    
    def predict(self, X):
        """Realiza predicciones con el modelo entrenado"""
        if self.model is None:
            raise ValueError("El modelo no está entrenado")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path):
        """Guarda el modelo en disco"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_info': self.model_info
        }
        joblib.dump(model_data, path)
        
    def load(self, path):
        """Carga el modelo desde disco"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_info = model_data['model_info']