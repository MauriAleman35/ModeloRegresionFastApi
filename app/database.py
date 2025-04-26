import os
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

# Obtener URI de MongoDB desde variables de entorno
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "EcommerML")

# Cliente de MongoDB
client = None

def get_database():
    """Retorna la conexión a la base de datos MongoDB"""
    global client
    if not client:
        client = MongoClient(MONGODB_URI)
    return client[DB_NAME]

def get_sales_data():
    """Obtiene los datos de ventas desde MongoDB y los convierte a DataFrame"""
    try:
        db = get_database()
        
        # Filtrar ventas (procesadas o confirmadas)
        filtro = {"estado": {"$in": ["Procesado", "Confirmado"]}}
        
        # Obtener documentos
        ventas = list(db.ventas.find(filtro))
        
        if not ventas or len(ventas) == 0:
            print("No se encontraron ventas que cumplan con los criterios")
            return None
        
        print(f"Se encontraron {len(ventas)} ventas válidas")
        
        # Convertir a DataFrame
        df = pd.DataFrame(ventas)
        
        # Asegurar que exista la columna createdAT
        if 'createdAT' not in df.columns:
            print("ADVERTENCIA: La colección no tiene columna 'createdAT'")
            return None
        
        # Convertir fechas a datetime y asignarla como columna fecha
        df['fecha'] = pd.to_datetime(df['createdAT'])
        
        # Extraer año y mes
        df['año'] = df['fecha'].dt.year
        df['mes'] = df['fecha'].dt.month
        
        # Agrupar por año y mes
        ventas_mensuales = df.groupby(['año', 'mes'])['total'].sum().reset_index()
        
        # Añadir nombre del mes
        meses = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio',
                 7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
        ventas_mensuales['nombre_mes'] = ventas_mensuales['mes'].map(meses)
        
        # Ordenar por año y mes
        ventas_mensuales = ventas_mensuales.sort_values(['año', 'mes'])
        
        # Exportar a CSV para referencia si se desea
        try:
            os.makedirs("data", exist_ok=True)
            ventas_mensuales.to_csv("data/ventas_from_mongodb.csv", index=False)
            print("Datos exportados a data/ventas_from_mongodb.csv para referencia")
        except:
            pass
        
        return ventas_mensuales
    
    except Exception as e:
        print(f"Error al obtener datos de ventas: {str(e)}")
        import traceback
        traceback.print_exc()
        return None