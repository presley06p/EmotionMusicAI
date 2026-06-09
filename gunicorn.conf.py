# gunicorn.conf.py — Production configuration

import multiprocessing

# Server socket
bind    = "0.0.0.0:8000"
backlog = 2048

# Workers
workers         = multiprocessing.cpu_count() * 2 + 1
worker_class    = "sync"
worker_connections = 1000
timeout         = 30
keepalive       = 2

# Logging
loglevel    = "info"
accesslog   = "-"   # stdout
errorlog    = "-"   # stderr

# Reload
reload      = False  # Set True only in dev
