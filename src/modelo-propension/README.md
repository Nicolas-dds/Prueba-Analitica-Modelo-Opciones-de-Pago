# Modelo de Propensión a Aceptación de Opciones de Pago

Solución E2E (MLOps) del modelo de propensión — Parte 1 del reto. Predice, por
obligación, la probabilidad de aceptar una opción de pago preaprobada en el mes
siguiente (RF-01 a RF-12).

> **Estado actual: implementado y entrenado con los datos reales de
> Bancolombia** (`data/`). Todo el pipeline (`ingest.py`, `features.py`,
> `split.py`, `train.py`, `explain.py`, `infer_batch.py`, `registry.py`,
> `monitor.py`) tiene lógica real, no solo firmas. Ya hay versiones
> registradas en `models/registry/` y un modelo campeón vigente
> (`models/registry/champion.json`); la API sirve ese campeón. Lo que queda
> pendiente para pasar de prototipo a producción real está en la sección
> [Qué falta para producción](#qué-falta-para-pasar-de-prototipo-a-producción-real).

## Arquitectura y flujo

Ver diagrama en la raíz del workspace:
[modelo_propension.excalidraw](../../modelo_propension.excalidraw).

Dos ciclos independientes, desacoplados a propósito:

- **Inferencia (obligatoria, cadencia mensual)**: siempre corre con el modelo
  campeón vigente, haya o no reentrenamiento ese mes.
- **Entrenamiento/reentrenamiento (bajo demanda)**: se dispara por alerta de
  `pipeline/monitor.py`, por una ventana de seguridad más espaciada, o
  manualmente. Nunca reemplaza al campeón automáticamente: siempre pasa por el
  gate de promoción (`pipeline/registry.promote`, métrica F1).

```
ingest -> features -> split -> train (tune+fit+evaluate OOT) -> registry.register_version -> registry.promote (gate)
                                                                                                        |
                                                                                                        v
                                                                                          champion.json (F1 >= campeón + min_improvement)
                                                                                                        |
                                                                                                        v
                                                        infer_batch (mensual, usa el campeón vigente) -> api/main.py -> Consumidores
                                                                                                        |
                                                                                                        v
                                                                                          monitor.py (drift, desempeño real) -> alerta -> reentrenar
```

## Estructura del proyecto

```
src/modelo-propension/
├── config/pipeline_config.yaml   # toda la configuración parametrizable (RNF-10)
├── config_loader.py               # carga el YAML
├── conftest.py                    # permite importar los paquetes sin instalar (carpeta con guion)
├── schemas/contracts.py           # contratos pydantic compartidos (pipeline + API)
├── pipeline/
│   ├── ingest.py                  # carga y valida datos crudos (RF-04, RF-05)
│   ├── features.py                # feature engineering, compartido train/infer (RF-05, RF-03)
│   ├── split.py                   # partición temporal train/val/OOT (RF-06)
│   ├── train.py                   # tuning, entrenamiento y evaluación OOT con XGBoost (RF-06)
│   ├── explain.py                 # explicabilidad SHAP global/local (RF-08)
│   ├── registry.py                # registry + gate de promoción F1 + rollback (RF-09, RF-10)
│   ├── infer_batch.py             # inferencia batch mensual (RF-07, RF-12)
│   └── monitor.py                 # drift (PSI/KS) y desempeño real vs esperado (RF-11)
├── orchestration/
│   ├── run_pipeline.py            # orquesta todo el entrenamiento + gate de CI/CD simulado (RF-10)
│   ├── run_monitoring.py          # job mensual de drift + disparo de reentrenamiento por alerta (RF-11)
│   └── generate_submission.py     # refit final sobre todo trtest + calificación de la OOT real (RF-07)
├── api/
│   ├── main.py                    # FastAPI: health, model/info, score, score/batch, model/promote — implementado
│   └── dependencies.py            # carga config + campeón vigente (sin caché de campeón)
├── tests/                         # pruebas estructurales (no requieren datos)
├── Dockerfile / .dockerignore
└── requirements.txt
```

## Setup del entorno (con `uv`)

```bash
cd src/modelo-propension
uv venv
uv pip install -r requirements.txt
```

## Comandos

### 1. Correr las pruebas estructurales
No requieren datos; validan imports, schemas, config y contratos de la API.

```bash
uv run pytest -v
```

### 2. Levantar el servicio de inferencia (FastAPI)
```bash
uv run uvicorn api.main:app --reload --port 8000
```

Endpoints disponibles:

| Método y ruta | Para qué sirve | Estado hoy (con modelo campeón registrado) |
|---|---|---|
| `GET /health` | Chequeo de disponibilidad (RNF-08) | `200 OK` |
| `GET /model/info` | Metadatos del campeón vigente (RF-08, RNF-04) | `200 OK` con la versión, métricas e hiperparámetros del campeón |
| `GET /score/{id_obligacion}` | Score batch más reciente de una obligación — endpoint que consume el Context Builder del sistema agéntico (RF-07, RF-12) | `200 OK` si ya se corrió `/score/batch` para algún mes; `404` si la obligación no aparece en el batch más reciente o no hay ningún batch todavía |
| `POST /score/batch?month=YYYY-MM` | Dispara la inferencia batch de ese mes (RF-07) | `200 OK` con la lista de scores; `501` si aún no hay modelo campeón registrado |
| `POST /model/promote?challenger_version=vXXXX` | Aplica el gate de promoción sobre una versión ya registrada (RF-10) | `200 OK` con el resultado del gate; `404` si la versión no existe en el registry |

Ejemplos con `curl`:
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/info
curl http://127.0.0.1:8000/score/OB-0001
curl -X POST "http://127.0.0.1:8000/score/batch?month=2026-09"
curl -X POST "http://127.0.0.1:8000/model/promote?challenger_version=v20260101000000"
```

> Nota sobre los tests: `tests/test_api_contracts.py` aísla la API contra un
> registry vacío en `tmp_path` (via monkeypatch), así que sus aserciones
> (`sin-modelo-registrado`, `501`) validan el contrato de "arranque en frío"
> sin importar si el repo ya tiene un campeón real registrado.

### 3. Entrenar / reentrenar (batch, no vía API)
```bash
uv run python -m orchestration.run_pipeline --month 2026-08
```
Encadena `ingest -> features -> split -> tune_hyperparameters -> train_model ->
evaluate_oot -> registry.register_version -> registry.promote`. El argumento
`--month` es el último mes con datos confiables (límite del OOT). Ya corrió
varias veces contra los datos reales de `data/` (ver `models/registry/`,
hay 4 versiones registradas y un campeón vigente en `champion.json`).

Imprime la versión promovida, o indica que el challenger no superó al campeón:
```
Version promovida: v20260901120000
# o bien:
El challenger no supero al campeon; no se promovio.
```

### 4. Monitoreo mensual (dispara reentrenamiento si hay alertas)
```bash
uv run python -m orchestration.run_monitoring --month 2026-08
```
Calcula drift de datos (PSI por variable, `pipeline.monitor.compute_data_drift`)
del mes contra `config.data.reference_month`; si `evaluate_alerts()` devuelve
alguna alerta (PSI > `config.monitoring.psi_threshold`), dispara
automáticamente `run_full_pipeline()` (mismo gate de promoción F1).

`pipeline/monitor.py` también implementa `compute_prediction_drift` (KS sobre
scores) y `compute_performance_drift` (F1 real vs. esperado), pero
`run_monitoring.py` todavía no los invoca: ambos requieren histórico real de
scores/etiquetas por mes (batches ya calificados con `/score/batch` + la
etiqueta observada en t+1), que hoy no se acumula en ningún lado. Integrarlos
implica: (1) persistir el batch calificado junto con la etiqueta real una vez
se observa, y (2) que `run_monitoring.py` lea ese histórico y llame a estas
dos funciones además de `compute_data_drift`.

### 5. Rollback manual
```python
from pipeline import registry
registry.rollback(registry_path="models/registry")
```
Revierte `champion.json` a la versión previa registrada en `history.json`.

### 6. Docker
Mismo image sirve para servir y para entrenar (solo cambia el comando):
```bash
docker build -t modelo-propension .

# Servir (contenedor siempre activo):
docker run -p 8000:8000 modelo-propension

# Entrenar (job puntual, sobreescribe el CMD):
docker run modelo-propension python -m orchestration.run_pipeline --month 2026-08
```

> **Producción real**: `registry.path` debe apuntar a almacenamiento
> compartido (no disco local del contenedor) si corres varias réplicas de la
> API o si el job de entrenamiento corre en un contenedor distinto — de lo
> contrario las réplicas de serving no verán al nuevo campeón.

## Configuración parametrizable

Todo vive en `config/pipeline_config.yaml`, nada debe quedar embebido en el
código (RNF-10):

| Sección | Qué controla |
|---|---|
| `data` | Ruta de datos crudos, columna id/label, mes de referencia para drift |
| `split` | Meses de corte train/val/OOT |
| `model` | Algoritmo (XGBoost), grilla de hiperparámetros, semilla |
| `registry` | Ruta del registry, métrica de promoción (`f1`), mejora mínima requerida |
| `monitoring` | Umbrales de PSI, KS y caída de desempeño |
| `api` | Host y puerto del servicio |

## Qué falta para pasar de prototipo a producción real

El pipeline completo (ingesta → features → split → train → explain →
infer_batch → registry → monitor) ya está implementado y corrido contra los
datos reales de Bancolombia. Lo pendiente es lo que separa a este prototipo
de un despliegue productivo real:

1. **Histórico de scores + etiqueta real por mes**: para cerrar
   `compute_prediction_drift` y `compute_performance_drift` en
   `run_monitoring.py` (ver sección de monitoreo mensual arriba) hace falta
   acumular, mes a mes, el batch calificado junto con la etiqueta observada
   en t+1 — hoy `infer_batch.py` solo persiste el score, no el histórico
   comparable contra el desempeño esperado.
2. **CI/CD real**: `orchestration/run_pipeline.py` simula el gate de
   promoción localmente; en producción esto debería vivir en un pipeline
   real (GitHub Actions, Jenkins, Airflow/Databricks Jobs) con ambientes
   separados (dev/staging/prod) y promoción entre ellos, no solo un script
   ejecutado a mano.
3. **Registry en almacenamiento compartido**: `pipeline/registry.py` escribe
   en disco local (`models/registry/`). Si la API corre con varias réplicas,
   o si el job de entrenamiento corre en un contenedor distinto al de
   serving, las réplicas no verán al nuevo campeón — `registry.path` debe
   apuntar a almacenamiento compartido (ej. S3/GCS/blob storage) antes de
   escalar horizontalmente.
4. **Caché del campeón en la API**: `api/dependencies.get_champion_metadata`
   relee el `champion.json` en cada request; conviene invalidar/refrescar el
   caché en vez de leer disco en cada llamada si el volumen de tráfico crece
   (RNF-07).
5. **Feature store real**: hoy `features.build_features` recalcula todo a
   partir de los 5 CSV crudos en cada corrida (train, infer_batch, monitor);
   en producción conviene materializar features versionadas en un feature
   store para evitar recomputar y garantizar consistencia entre consumidores.
