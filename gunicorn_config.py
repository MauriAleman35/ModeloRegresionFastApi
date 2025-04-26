# Configuración para Gunicorn en producción
import os

# Usar el worker de Uvicorn (ASGI) en lugar del worker por defecto (WSGI)
worker_class = "uvicorn.workers.UvicornWorker"

# Configuración general
workers = 1  # Para aplicaciones de ML, un worker suele ser suficiente
threads = 2
timeout = 120  # Mayor timeout para procesos de ML que pueden tardar
max_requests = 1000
max_requests_jitter = 50

# Puerto y host
bind = "0.0.0.0:" + os.environ.get("PORT", "8000")

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"