# Configuración para Gunicorn en producción
workers = 1  # Para aplicaciones de ML, un worker suele ser suficiente
threads = 2
timeout = 120  # Mayor timeout para procesos de ML que pueden tardar
max_requests = 1000
max_requests_jitter = 50