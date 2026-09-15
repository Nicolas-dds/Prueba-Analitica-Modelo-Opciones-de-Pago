"""Permite importar los modulos del paquete sin instalarlo (la carpeta contenedora usa guion)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
