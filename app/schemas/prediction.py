from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional

class PredictionMonthResult(BaseModel):
    mes: int
    año: int
    nombre_mes: str
    prediccion_pesimista: float
    prediccion_esperada: float
    prediccion_optimista: float

class PredictionRequest(BaseModel):
    meses: int = Field(..., ge=1, le=24, description="Número de meses a predecir (entre 1 y 24)")
    
    @validator('meses')
    def validate_meses(cls, v):
        if v < 1:
            raise ValueError('El número de meses debe ser al menos 1')
        if v > 24:
            raise ValueError('El número máximo de meses a predecir es 24')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "meses": 6
            }
        }

class PredictionResponse(BaseModel):
    predicciones: List[PredictionMonthResult]
    total_esperado: float
    meses_predichos: int

class ModelInfoResponse(BaseModel):
    model_name: str
    r2_score: float
    rmse: float
    data_points: int
    last_trained: Optional[str] = None
    coeficientes: Dict[str, float] = {}