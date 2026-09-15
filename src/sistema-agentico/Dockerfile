# Imagen oficial de `uv` (Astral) con Python 3.12 preinstalado, coherente con
# `requires-python = ">=3.12"` en pyproject.toml. Evita instalar uv manualmente
# (pip install uv / copiar el binario desde otra imagen) y evita fijar una versión
# de Python distinta a la usada en desarrollo (.python-version=3.14, pero
# requires-python ya permite 3.12+; se fija el 3.12 en la imagen para máxima
# compatibilidad de wheels prebuilt entre dependencias).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copiar sólo los manifiestos primero para maximizar el cacheo de capas de Docker:
# `uv sync` sólo se re-ejecuta cuando `pyproject.toml`/`uv.lock` cambian, no en cada
# cambio de código fuente.
COPY pyproject.toml uv.lock ./

# `--frozen`: usa exactamente `uv.lock` sin resolver/actualizar dependencias.
# `--no-dev`: `pyproject.toml` no define un grupo `dev` separado (pytest/hypothesis
# están en las dependencias principales), por lo que este flag no excluye nada hoy,
# pero se mantiene por convención/robustez ante cambios futuros del pyproject.
RUN uv sync --frozen --no-dev

# Resto del código fuente (ver .dockerignore para lo que se excluye: tests/, .venv/,
# caches, .env).
COPY sistema_agentico/ ./sistema_agentico/
COPY config/ ./config/
COPY conftest.py run_demo.py ./

# `sistema_agentico/` queda directamente bajo `/app`, por lo que
# `PYTHONPATH=/app` permite `import sistema_agentico` sin instalar el paquete
# (consistente con `[tool.uv] package = false` en pyproject.toml y con la
# convención de modelo-propension/Dockerfile).
ENV PYTHONPATH=/app

# Puerto de sistema-agentico (distinto al 8000 de modelo-propension), igual al
# default de SISTEMA_AGENTICO_PORT en settings.py.
EXPOSE 8001

# `uv run` ejecuta dentro del entorno virtual creado por `uv sync` (.venv/ local al
# proyecto) sin necesidad de activarlo explícitamente ni de instalar el paquete.
CMD ["uv", "run", "uvicorn", "sistema_agentico.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
