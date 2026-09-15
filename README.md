# AI Prod — Modelo de Propensión + Sistema Agéntico de Cobranza

Solución E2E para la prueba de Bancolombia: un
modelo de propensión a aceptación de opciones de pago (`src/modelo-propension`)
y un sistema multiagente de gestión de cartera en mora (`src/sistema-agentico`),
orquestados juntos con `docker-compose.yml`.

| Servicio | Puerto | Rol |
|---|---|---|
| `modelo-propension` | `8000` | API de scoring del modelo (FastAPI) |
| `sistema-agentico` | `8001` | API del sistema de agentes (FastAPI), consume a `modelo-propension` |

> **Seguridad**: ninguno de los dos servicios expone autenticación ni
> autorización. Es un prototipo para uso local/demo; un despliegue real
> requeriría, como mínimo, auth y restringir el acceso de red a estos puertos.

## Prerrequisitos

- Docker y Docker Compose instalados.
- Los archivos de datos crudos (**no incluidos en este repositorio**, ver
  siguiente sección).
- Un archivo `.env` propio en la raíz del proyecto (**tampoco incluido**, ver
  siguiente sección).

## 1. Datos

La carpeta `data/` no se sube al repositorio. Antes de levantar el proyecto,
crea `data/` en la raíz y coloca ahí los archivos entregados para la prueba:

```
data/
├── prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv
├── prueba_op_base_pivot_var_rpta_alt_enmascarado_oot.csv
├── prueba_op_maestra_cuotas_pagos_mes_hist_enmascarado_completa.csv
├── prueba_op_master_customer_data_enmascarado_completa.csv
├── prueba_op_probabilidad_oblig_base_hist_enmascarado_completa.csv
└── sample_submission.csv
```

Esta carpeta se monta de solo lectura en `sistema-agentico` (usada solo si
`USE_REAL_PROFILE=1`). El modelo campeón ya viene entrenado y versionado en
`src/modelo-propension/models/registry/`, así que **no es obligatorio tener
`data/` para levantar los servicios con docker-compose** salvo que quieras
reentrenar o activar `USE_REAL_PROFILE`.

## 2. Variables de entorno (`.env`)

Crea un archivo `.env` en la raíz del proyecto (mismo nivel que
`docker-compose.yml`) con estas variables:

```bash
# Clave real de OpenAI. Solo se usa si USE_REAL_LLM=1.
OPENAI_API_KEY=

# 1 = el agente conversacional llama a OpenAI de verdad.
# 0 (default) = usa un chat model fake determinista, sin llamadas reales.
USE_REAL_LLM=0

# 1 = sistema-agentico llama al servicio modelo-propension real vía Docker Compose.
# 0 (default) = usa un cliente de propensión fake en memoria.
USE_REAL_PROPENSION=1

# 1 = usa el perfil/historial real derivado de data/..._trtest.csv (requiere data/).
# 0 (default) = usa un perfil sintético en memoria.
USE_REAL_PROFILE=0
```

Si `USE_REAL_LLM=1` sin una `OPENAI_API_KEY` válida, `sistema-agentico` falla
al iniciar con un error explícito (no hay fallback silencioso).

## 3. Levantar el proyecto

Desde la raíz del proyecto:

```bash
docker compose up --build
```

Esto construye e inicia ambos servicios. `sistema-agentico` espera a que
`modelo-propension` esté saludable (`healthcheck`) antes de arrancar, y lo
alcanza en la red interna de Compose como `http://modelo-propension:8000`.

## 4. Verificar que todo funciona

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

### Endpoints de `modelo-propension` (puerto 8000)

| Método y ruta | Para qué sirve |
|---|---|
| `GET /health` | Chequeo de disponibilidad |
| `GET /model/info` | Metadatos del modelo campeón vigente (versión, métricas, hiperparámetros) |
| `GET /score/{id_obligacion}` | Score batch más reciente de una obligación puntual |
| `POST /score/batch?month=YYYY-MM` | Dispara la inferencia batch para ese mes |
| `POST /model/promote?challenger_version=vXXXX` | Aplica el gate de promoción sobre una versión ya registrada del modelo |

```bash
curl http://localhost:8000/model/info
curl http://localhost:8000/score/OB-0001
curl -X POST "http://localhost:8000/score/batch?month=2026-09"
curl -X POST "http://localhost:8000/model/promote?challenger_version=v20260913073248"
```

Entrenamiento, reentrenamiento, monitoreo y rollback del modelo son procesos
batch aparte, fuera de `docker compose up` — ver
[`src/modelo-propension/README.md`](./src/modelo-propension/README.md).

### Endpoints de `sistema-agentico` (puerto 8001)

`sistema-agentico` opera en dos modos (RF-13):

### Modo reactivo

El cliente contacta para consultar deuda, negociar o reportar dificultad.
Procesa una sola interacción de punta a punta:

```bash
curl -X POST http://localhost:8001/interacciones/reactiva \
  -H "Content-Type: application/json" \
  -d '{
    "id_obligacion": "SYN-OBL-00000",
    "mensaje_cliente": "Hola, quisiera saber qué opciones tengo para pagar mi deuda"
  }'
```

Opcionalmente se puede enviar el historial de la conversación para dar
contexto al agente:

```bash
curl -X POST http://localhost:8001/interacciones/reactiva \
  -H "Content-Type: application/json" \
  -d '{
    "id_obligacion": "SYN-OBL-00000",
    "mensaje_cliente": "¿Y si no puedo pagar eso, hay otra alternativa?",
    "historial_conversacion": ["Hola, quisiera saber qué opciones tengo para pagar mi deuda"]
  }'
```

### Modo proactivo

El sistema inicia la gestión sobre un lote de obligaciones. La lista de ids
se reordena internamente por score de propensión (mayor a menor) antes de
procesarse, así que el orden de la respuesta puede diferir del orden enviado:

```bash
curl -X POST http://localhost:8001/interacciones/proactiva \
  -H "Content-Type: application/json" \
  -d '{"ids_obligacion": ["SYN-OBL-00001", "SYN-OBL-00002", "SYN-OBL-00003"]}'
```

La respuesta es una lista con un resultado por obligación
(`completado`/`resultado`/`error`); una falla en una obligación no detiene el
resto del lote.

### Escalamiento a gestor humano

En ambos modos, la respuesta puede venir con `"status": "escalated"` en vez
de `"final"` : ocurre ante solicitudes sensibles, intentos de
manipulación, información contradictoria, o fallos técnicos. En ese caso
`respuesta` viene en `null` y `escalation_reason` indica el motivo:

```json
{
  "trace_id": "...",
  "status": "escalated",
  "respuesta": null,
  "escalation_reason": "manipulacion_detectada",
  "trace_logged": true
}
```

### Trazabilidad

Cada interacción (reactiva o proactiva) queda registrada de forma auditable
e inmutable en un log JSONL append-only, dentro del contenedor de
`sistema-agentico`, en `data_output/trace_log_demo.jsonl`. El campo
`trace_logged` de cada respuesta indica si el registro se guardó
correctamente.

Para detener y limpiar los contenedores:

```bash
docker compose down
```
