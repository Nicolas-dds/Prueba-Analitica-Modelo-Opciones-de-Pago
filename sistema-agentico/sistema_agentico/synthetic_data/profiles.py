"""Generador de perfiles sintéticos de cliente/obligación (Requirement 13.1, 13.2).

Ver Requirement 13 de `requirements.md` ("Datos sintéticos para pruebas") y Component 3
(`ClienteObligacionContext`) / Component 4 (`OfertaAplicada`) de `design.md`.

Este módulo genera instancias sintéticas de `ClienteObligacionContext` (y sus building
blocks: `ScoreInfo`, `OfertaAplicada`) para alimentar pruebas y el dataset dorado (tarea 14),
sin depender de ni copiar ningún registro real. Las distribuciones de valores (rangos de
`dias_mora`, `exposicion`, `score`, categorías de `segmento`/`producto`) se calibraron
observando el *esquema* (nombres de columnas y rangos aproximados) de los archivos
`data/prueba_op_*` (ya enmascarados por el banco) y `data/Metadata.xlsx`, nunca copiando
valores identificatorios ni filas completas (RF-23).

Marca explícita de dato sintético (Requirement 13.2)
------------------------------------------------------
Ninguno de los dataclasses core (`ClienteObligacionContext`, etc., definidos en `types.py`)
tiene un campo dedicado `is_synthetic`; no se modifican esos contratos aquí (fueron fijados
en la tarea 1 y son consumidos por todo el pipeline). En su lugar, este módulo usa una
**convención de prefijo** sobre los identificadores, aplicada de forma consistente:

- `id_cliente` sintético siempre inicia con el prefijo `SYN-CLI-` (p.ej. `SYN-CLI-00042`).
- `id_obligacion` sintético siempre inicia con el prefijo `SYN-OBL-` (p.ej. `SYN-OBL-00042`).
- `version_modelo` en el `ScoreInfo` sintético siempre inicia con `SYN-` (p.ej. `SYN-model-v0`).

Cualquier componente puede distinguir un registro sintético de uno real comprobando
`id_cliente.startswith(SYNTHETIC_ID_PREFIX_CLIENTE)` (o el helper `is_synthetic_id`).
Esta misma convención debe reutilizarse en las tareas 4.2 (conversaciones sintéticas),
4.3 (tests) y 14.1 (escenarios dorados) para mantener consistencia en todo el módulo.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from sistema_agentico.types import ClienteObligacionContext, OfertaAplicada, ScoreInfo

# ---------------------------------------------------------------------------
# Convención de marca de dato sintético (Requirement 13.2)
# ---------------------------------------------------------------------------

SYNTHETIC_ID_PREFIX_CLIENTE = "SYN-CLI-"
SYNTHETIC_ID_PREFIX_OBLIGACION = "SYN-OBL-"
SYNTHETIC_MODEL_VERSION_PREFIX = "SYN-"


def is_synthetic_id(identificador: str) -> bool:
    """Retorna `True` si `identificador` fue generado por este módulo (RF-23).

    Reconoce tanto el prefijo de cliente como el de obligación, de forma que pueda
    aplicarse indistintamente sobre `id_cliente` o `id_obligacion`.
    """
    return identificador.startswith(SYNTHETIC_ID_PREFIX_CLIENTE) or identificador.startswith(
        SYNTHETIC_ID_PREFIX_OBLIGACION
    )


def is_synthetic_context(contexto: ClienteObligacionContext) -> bool:
    """Retorna `True` si `contexto` fue generado sintéticamente (marca de ambos ids)."""
    return contexto.id_cliente.startswith(SYNTHETIC_ID_PREFIX_CLIENTE) and (
        contexto.id_obligacion.startswith(SYNTHETIC_ID_PREFIX_OBLIGACION)
    )


# ---------------------------------------------------------------------------
# Distribuciones de referencia derivadas del ESQUEMA de `data/prueba_op_*`
# (columnas y rangos observados; nunca valores/filas reales) y `data/Metadata.xlsx`.
# ---------------------------------------------------------------------------

# Segmentos observados en `prueba_op_base_pivot_var_rpta_alt_*` (columna `segmento`).
SEGMENTOS = ["Personal", "Personal plus", "Preferencial", "Social", "Micropyme", "Pymes"]

# Productos observados en la misma fuente (columna `producto`), acotados a un subconjunto
# representativo relevante para cobranza de cartera de consumo.
PRODUCTOS = [
    "LIBRE INVERSION",
    "CREDITOS DE CONSUMO",
    "TARJETA DE CREDITO",
    "LIBRANZA",
    "CREDIAGIL",
]

# Canales de gestión de cobranza típicos (no presentes literalmente en el esquema fuente;
# son categorías de negocio razonables para `canal_gestion`, campo no derivado de PII).
CANALES_GESTION = ["call_center", "sms", "whatsapp", "email", "app_movil"]

# Tipos de opción de pago, alineados a `COOLDOWN_MESES_POR_TIPO` de `design.md`.
TIPOS_OPCION_PAGO = ["ampliacion_plazo", "reduccion_cuota", "renegociacion_tasa", "reestructuracion"]
COOLDOWN_MESES_POR_TIPO = {
    "ampliacion_plazo": 3,
    "reduccion_cuota": 3,
    "renegociacion_tasa": 4,
    "reestructuracion": 4,
}

# Rango de `dias_mora` observado en `prueba_op_base_pivot_var_rpta_alt_*` (`dias_mora_fin`:
# 0 a ~372 en la muestra); se acota el límite superior para representar mora temprana/media,
# consistente con el alcance del reto (RNF-15).
DIAS_MORA_MIN = 0
DIAS_MORA_MAX = 180

# Rango de `exposicion` (equivalente a `vlr_obligacion`/`saldo_capital` de la fuente),
# expresado en pesos colombianos, acotado a un rango representativo de cartera de consumo.
EXPOSICION_MIN = 200_000.0
EXPOSICION_MAX = 50_000_000.0

VERSION_MODELO_SINTETICA = f"{SYNTHETIC_MODEL_VERSION_PREFIX}model-v0"


@dataclass(frozen=True)
class SyntheticProfileOptions:
    """Parámetros opcionales para sesgar la generación de un perfil individual.

    Todos los campos son opcionales; si no se especifican, `generate_profile` elige
    valores aleatorios (pero deterministas dado el `seed`/`rng`) dentro de los rangos
    de referencia. Permiten construir escenarios dirigidos (p.ej. "en cooldown",
    "en el límite de 3 opciones del mes") reutilizables por el dataset dorado (tarea 14)
    y por las pruebas del `EligibilityEngine` (tarea 6).
    """

    dias_mora: int | None = None
    exposicion: float | None = None
    segmento: str | None = None
    canal_gestion: str | None = None
    ofertas_aplicadas_mes: list[OfertaAplicada] | None = None
    ultima_opcion_aplicada: OfertaAplicada | None = None
    acuerdo_pago_vigente: bool | None = None
    restriccion_vigente: bool | None = None
    score: float | None = None
    score_degradado: bool | None = None


def _resolve_rng(seed: int | random.Random | None) -> random.Random:
    """Normaliza `seed` a una instancia de `random.Random` reutilizable y determinista.

    Acepta un `int` (nueva instancia sembrada con ese valor), una instancia ya existente
    de `random.Random` (se reutiliza tal cual, permitiendo compartir estado entre varias
    llamadas), o `None` (nueva instancia sin semilla fija, no reproducible).
    """
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


def _fecha_aplicacion_hace_meses(rng: random.Random, meses: float) -> str:
    """Retorna una fecha ISO (`YYYY-MM-DD`) aproximadamente `meses` meses antes de hoy.

    Usa una aproximación de 30 días por mes, suficiente para escenarios de prueba
    (no se requiere precisión calendario exacta para generar datos sintéticos).
    """
    from datetime import date, timedelta

    dias = int(meses * 30) + rng.randint(-5, 5)
    fecha = date.today() - timedelta(days=max(dias, 0))
    return fecha.isoformat()


def generate_oferta_aplicada(
    seed: int | random.Random | None = None,
    *,
    tipo_opcion: str | None = None,
    meses_desde_aplicacion: float | None = None,
) -> OfertaAplicada:
    """Genera una `OfertaAplicada` sintética individual.

    Si `meses_desde_aplicacion` no se especifica, se elige uniformemente entre 0 y 6
    meses atrás, cubriendo tanto escenarios dentro de cooldown como fuera de él.
    """
    rng = _resolve_rng(seed)
    tipo = tipo_opcion or rng.choice(TIPOS_OPCION_PAGO)
    meses = meses_desde_aplicacion if meses_desde_aplicacion is not None else rng.uniform(0, 6)
    return OfertaAplicada(
        tipo_opcion=tipo,
        fecha_aplicacion=_fecha_aplicacion_hace_meses(rng, meses),
        cooldown_meses=COOLDOWN_MESES_POR_TIPO[tipo],
    )


def generate_score_info(seed: int | random.Random | None = None, *, score: float | None = None, degradado: bool | None = None) -> ScoreInfo:
    """Genera un `ScoreInfo` sintético con `version_modelo` marcada (`SYN-` prefix)."""
    rng = _resolve_rng(seed)
    valor_score = score if score is not None else round(rng.uniform(0.0, 1.0), 4)
    decil = min(10, max(1, int(valor_score * 10) + 1))
    return ScoreInfo(
        score=valor_score,
        decil=decil,
        version_modelo=VERSION_MODELO_SINTETICA,
        fecha_calificacion=_fecha_aplicacion_hace_meses(rng, 0),
        degradado=degradado if degradado is not None else False,
    )


def generate_profile(
    seed: int | random.Random | None = None,
    *,
    index: int | None = None,
    options: SyntheticProfileOptions | None = None,
) -> ClienteObligacionContext:
    """Genera un `ClienteObligacionContext` sintético completo, determinista dado `seed`.

    Args:
        seed: entero para sembrar un `random.Random` nuevo, una instancia de
            `random.Random` ya existente (reutilizada, útil para generar lotes
            reproducibles con `generate_profiles`), o `None` (no reproducible).
        index: entero usado para componer un `id_cliente`/`id_obligacion` únicos y
            legibles (p.ej. `SYN-CLI-00007`). Si es `None`, se usa un entero aleatorio
            derivado de `rng`, lo que puede producir colisiones bajo uso masivo sin
            `index` explícito; para lotes, `generate_profiles` siempre pasa un `index`
            secuencial.
        options: overrides opcionales para sesgar campos concretos (ver
            `SyntheticProfileOptions`), útiles para construir escenarios dirigidos
            (p.ej. cooldown activo, tope de 3 opciones/mes, acuerdo vigente).

    Returns:
        Un `ClienteObligacionContext` cuyos `id_cliente`/`id_obligacion` llevan el
        prefijo sintético (Requirement 13.2) y que no contiene ningún dato personal
        real (Requirement 13.1): los identificadores son secuenciales/sintéticos, no
        nombres, cédulas ni cuentas.
    """
    rng = _resolve_rng(seed)
    opts = options or SyntheticProfileOptions()
    numero = index if index is not None else rng.randint(0, 999_999)
    sufijo = f"{numero:05d}"

    dias_mora = opts.dias_mora if opts.dias_mora is not None else rng.randint(DIAS_MORA_MIN, DIAS_MORA_MAX)
    exposicion = (
        opts.exposicion if opts.exposicion is not None else round(rng.uniform(EXPOSICION_MIN, EXPOSICION_MAX), 2)
    )
    segmento = opts.segmento if opts.segmento is not None else rng.choice(SEGMENTOS)
    canal_gestion = opts.canal_gestion if opts.canal_gestion is not None else rng.choice(CANALES_GESTION)

    ofertas_aplicadas_mes = (
        opts.ofertas_aplicadas_mes
        if opts.ofertas_aplicadas_mes is not None
        else _generar_ofertas_aplicadas_mes(rng)
    )
    ultima_opcion_aplicada = (
        opts.ultima_opcion_aplicada
        if opts.ultima_opcion_aplicada is not None
        else (ofertas_aplicadas_mes[-1] if ofertas_aplicadas_mes else None)
    )
    acuerdo_pago_vigente = opts.acuerdo_pago_vigente if opts.acuerdo_pago_vigente is not None else rng.random() < 0.15
    restriccion_vigente = opts.restriccion_vigente if opts.restriccion_vigente is not None else rng.random() < 0.05

    score = generate_score_info(rng, score=opts.score, degradado=opts.score_degradado)

    return ClienteObligacionContext(
        id_obligacion=f"{SYNTHETIC_ID_PREFIX_OBLIGACION}{sufijo}",
        id_cliente=f"{SYNTHETIC_ID_PREFIX_CLIENTE}{sufijo}",
        dias_mora=dias_mora,
        exposicion=exposicion,
        segmento=segmento,
        canal_gestion=canal_gestion,
        score=score,
        ofertas_aplicadas_mes=ofertas_aplicadas_mes,
        ultima_opcion_aplicada=ultima_opcion_aplicada,
        acuerdo_pago_vigente=acuerdo_pago_vigente,
        restriccion_vigente=restriccion_vigente,
    )


def _generar_ofertas_aplicadas_mes(rng: random.Random) -> list[OfertaAplicada]:
    """Genera una lista realista de `OfertaAplicada` para el mes en curso.

    Distribución deliberada para poder ejercitar el `EligibilityEngine` (tarea 6) con
    escenarios variados: la mayoría de perfiles sin ofertas, algunos con 1-2, y una
    fracción menor exactamente en el tope de 3 (RF-15 / Property 8).
    """
    peso_cantidad = rng.choices([0, 1, 2, 3], weights=[0.5, 0.25, 0.15, 0.10], k=1)[0]
    ofertas: list[OfertaAplicada] = []
    for _ in range(peso_cantidad):
        # meses_desde_aplicacion en [0, 1) para reflejar que son ofertas "del mes".
        ofertas.append(generate_oferta_aplicada(rng, meses_desde_aplicacion=rng.uniform(0.0, 0.9)))
    return ofertas


def generate_cooldown_scenarios(seed: int | random.Random | None = None) -> list[ClienteObligacionContext]:
    """Genera un conjunto fijo de perfiles que cubren los escenarios de cooldown clave.

    Pensado para alimentar directamente pruebas del `EligibilityEngine` (tarea 6) y el
    dataset dorado (tarea 14), sin depender de la aleatoriedad general de
    `generate_profiles`. Cubre, para cada `tipo_opcion`:

    - Sin ofertas aplicadas (elegible por defecto).
    - Última oferta aplicada dentro de la ventana de cooldown (no elegible para ese tipo).
    - Última oferta aplicada justo fuera de la ventana de cooldown (elegible de nuevo).
    - Exactamente 3 ofertas aplicadas en el mes (tope alcanzado, sin importar cooldown).
    """
    rng = _resolve_rng(seed)
    perfiles: list[ClienteObligacionContext] = []
    contador = 0

    for tipo, cooldown_meses in COOLDOWN_MESES_POR_TIPO.items():
        # Sin ofertas aplicadas.
        perfiles.append(
            generate_profile(
                rng,
                index=contador,
                options=SyntheticProfileOptions(ofertas_aplicadas_mes=[], ultima_opcion_aplicada=None),
            )
        )
        contador += 1

        # Dentro de cooldown (a mitad de la ventana).
        oferta_dentro = generate_oferta_aplicada(rng, tipo_opcion=tipo, meses_desde_aplicacion=cooldown_meses / 2)
        perfiles.append(
            generate_profile(
                rng,
                index=contador,
                options=SyntheticProfileOptions(
                    ofertas_aplicadas_mes=[oferta_dentro], ultima_opcion_aplicada=oferta_dentro
                ),
            )
        )
        contador += 1

        # Justo fuera de cooldown (un mes más que la ventana).
        oferta_fuera = generate_oferta_aplicada(rng, tipo_opcion=tipo, meses_desde_aplicacion=cooldown_meses + 1)
        perfiles.append(
            generate_profile(
                rng,
                index=contador,
                options=SyntheticProfileOptions(
                    ofertas_aplicadas_mes=[oferta_fuera], ultima_opcion_aplicada=oferta_fuera
                ),
            )
        )
        contador += 1

    # Tope de 3 opciones aplicadas en el mes (independiente del tipo).
    tope_ofertas = [
        generate_oferta_aplicada(rng, meses_desde_aplicacion=rng.uniform(0.0, 0.5)) for _ in range(3)
    ]
    perfiles.append(
        generate_profile(
            rng,
            index=contador,
            options=SyntheticProfileOptions(
                ofertas_aplicadas_mes=tope_ofertas, ultima_opcion_aplicada=tope_ofertas[-1]
            ),
        )
    )

    return perfiles


def generate_profiles(n: int, seed: int | random.Random | None = None) -> list[ClienteObligacionContext]:
    """Genera `n` perfiles sintéticos deterministas, con `id_cliente`/`id_obligacion` únicos.

    Reutiliza una única instancia de `random.Random` (derivada de `seed`) a través de
    todas las llamadas a `generate_profile`, garantizando que el mismo `seed` produzca
    siempre exactamente la misma lista de perfiles (reproducibilidad, RNF-06).
    """
    rng = _resolve_rng(seed)
    return [generate_profile(rng, index=i) for i in range(n)]
