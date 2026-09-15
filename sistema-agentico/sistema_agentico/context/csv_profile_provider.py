"""`CsvObligacionProfileProvider`: perfil/historial real derivado de `data/prueba_op_*` (tarea 19).

Extensión de integración de datos reales solicitada explícitamente por el usuario, fuera
del alcance original de `requirements.md`/`design.md` (ver tarea 19 de
`.kiro/specs/sistema-agentico-cobranza/tasks.md`). El resto del sistema (`ContextBuilder`,
`EligibilityEngine`, etc.) no cambia: este módulo solo aporta una implementación adicional,
estructuralmente compatible, de `ObligacionProfileProvider` (Protocol definido en
`context_builder.py`), inyectable en lugar de `InMemoryObligacionProfileProvider`.

Fuente de datos y derivación de `id_obligacion`
------------------------------------------------
`modelo-propension` (`pipeline/ingest.py`) construye `id_obligacion` concatenando
`nit_enmascarado#num_oblig_orig_enmascarado#num_oblig_enmascarado`. Ese mismo formato es
el que usa `PropensionClient.get_score` (contrato `GET /score/{id_obligacion}`), así que
reimplementamos aquí exactamente la misma lógica de concatenación (sin importar código de
`modelo-propension`: son dos proyectos/servicios independientes, sin dependencia cruzada
en Python).

Solo el archivo `trtest`
(`prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv`) trae, por fila
obligación-mes, los campos de perfil/historial que este provider necesita
(`dias_mora_fin`, `vlr_vencido`, `vlr_obligacion`, `segmento`, `aplicativo`,
`marca_alt_apli`, `alternativa_aplicada_agr`, `cant_acuerdo_binario`). El archivo `oot`
(el que se califica vía `/score`) NO trae estas columnas — solo identificadores y
`fecha_var_rpta_alt` —, así que una obligación puede tener score real sin tener perfil
real: es una limitación real de los datos (ver `features.py`, ~49% de cobertura de la OOT
real en `trtest`), no de este provider. Para el resto, `get_profile` lanza
`ObligacionProfileNotFoundError` y `ContextBuilder` lo trata como "perfil ausente"
(Requirement 3.5, comportamiento ya implementado) en lugar de fabricar datos.

Cuando un `id_obligacion` tiene varias filas en `trtest` (varios meses de
`fecha_var_rpta_alt`), se usa la fila MÁS RECIENTE (máximo `fecha_var_rpta_alt`) como
snapshot del perfil.

Mapeo de campos (best-effort, ambigüedades documentadas explícitamente)
------------------------------------------------------------------------
- `id_cliente` <- `nit_enmascarado` (como texto).
- `dias_mora` <- `dias_mora_fin`, casteado a `int`. Si es `NaN` se deja `None` (NO se
  asume 0): `dias_mora=0` tiene significado de negocio real (obligación al día), así que
  confundirlo con "dato ausente" sería un error, no una simplificación razonable.
- `exposicion` <- `vlr_vencido` si está presente/no nulo; si no, se usa `vlr_obligacion`
  como respaldo. `vlr_vencido` (valor vencido/en mora) es conceptualmente más cercano a
  "exposición en riesgo" que `vlr_obligacion` (valor desembolsado/cupo original de la
  obligación, que no cambia mes a mes); se usa como respaldo razonable únicamente cuando
  el primero falta.
- `segmento` <- `segmento` (columna homónima).
- `canal_gestion` <- `aplicativo`. Es el análogo más cercano en el esquema real a "canal
  de gestión" (aplicativo desde el que se gestiona la obligación); `producto` es un
  concepto distinto (tipo de producto financiero, no canal), por lo que no se usa aquí.
- `ofertas_aplicadas_mes` / `ultima_opcion_aplicada`: `trtest` no tiene un historial
  limpio de aplicación por tipo de oferta que calce 1:1 con el catálogo de
  `sistema_agentico.eligibility` (`ampliacion_plazo`/`reduccion_cuota`/
  `renegociacion_tasa`/`reestructuracion`). Se usa una heurística conservadora: solo si
  `marca_alt_apli == "SI"` (hubo aplicación efectiva) Y `alternativa_aplicada_agr` tiene
  un valor mapeable con confianza (ver `_ALTERNATIVA_AGR_A_TIPO_OPCION` abajo) se
  construye una `OfertaAplicada`; si no, se deja `ofertas_aplicadas_mes=[]` (lista vacía,
  NO `None`) en vez de adivinar. La distinción `[]` vs `None` importa para
  `ContextBuilder._detectar_contradiccion`: `None` se trataría como "campo ausente"
  (severo), mientras que `[]` es "sin historial de ofertas conocido" (caso normal,
  perfectamente válido). `fecha_aplicacion` se deriva de `fecha_var_rpta_alt` (entero
  `YYYYMM`, mismo formato que usa `ingest.py` para su `PeriodIndex`) como el primer día
  de ese mes; `cooldown_meses` usa los mismos valores por defecto que
  `config/reglas_elegibilidad.yaml` (ver `_COOLDOWN_MESES_POR_TIPO`).
- `ultima_opcion_aplicada`: la única entrada de `ofertas_aplicadas_mes` si existe (la fila
  más reciente solo puede aportar como máximo una), o `None`.
- `acuerdo_pago_vigente`: `True` únicamente cuando HAY una señal doblemente confirmada:
  `cant_acuerdo_binario == 1` Y además existe una `ultima_opcion_aplicada` mapeada que la
  sustente. Se exige esta doble condición a propósito: `ContextBuilder` ya trata
  `acuerdo_pago_vigente=True` sin `ultima_opcion_aplicada` como una contradicción
  (Requirement 3.5) — exigir ambas señales aquí evita generar esa contradicción por una
  heurística demasiado optimista. Ante cualquier ambigüedad se prefiere `False`: un falso
  negativo (no marcar un acuerdo vigente real) es más seguro que un falso positivo (que
  suprimiría indebidamente la elegibilidad de acuerdo de pago).
- `restriccion_vigente`: sin análogo claro en el esquema real de `trtest`. Se deja
  siempre en `False` — brecha de datos conocida y documentada explícitamente, no una
  señal fabricada.

Fail-fast en la carga del CSV
-------------------------------
La carga del archivo `trtest` ocurre en el constructor (carga eager, no perezosa): si el
archivo no existe o no puede leerse, se lanza `RawDataUnavailableError` de inmediato en
vez de crear un provider "vacío" que fallaría silenciosamente en cada `get_profile`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sistema_agentico.context.context_builder import ObligacionProfileData, ObligacionProfileNotFoundError
from sistema_agentico.types import OfertaAplicada

__all__ = [
    "CsvObligacionProfileProvider",
    "RawDataUnavailableError",
    "TRTEST_FILENAME",
]

# Nombre del archivo crudo de `modelo-propension` que trae los campos de perfil/historial
# (ver docstring del módulo). Coincide con `config["data"]["trtest_file"]` en
# `src/modelo-propension/config/pipeline_config.yaml`.
TRTEST_FILENAME = "prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv"

# Mismas columnas identificadoras que `ingest._build_id_obligacion` (modelo-propension).
_ID_COMPONENT_COLUMNS = ["nit_enmascarado", "num_oblig_orig_enmascarado", "num_oblig_enmascarado"]

# Columnas mínimas necesarias del CSV para derivar `ObligacionProfileData` (ver mapeo en
# el docstring del módulo). Restringir las columnas leídas evita cargar en memoria las
# ~49 columnas del archivo real cuando solo se necesitan estas.
_USECOLS = [
    *_ID_COMPONENT_COLUMNS,
    "fecha_var_rpta_alt",
    "dias_mora_fin",
    "vlr_vencido",
    "vlr_obligacion",
    "segmento",
    "aplicativo",
    "marca_alt_apli",
    "alternativa_aplicada_agr",
    "cant_acuerdo_binario",
]

# Mapeo best-effort de `alternativa_aplicada_agr` (valores reales observados en
# `trtest`) al catálogo de `tipo_opcion` de `sistema_agentico.eligibility`
# (`ampliacion_plazo`/`reduccion_cuota`/`renegociacion_tasa`/`reestructuracion`).
#
# - "AMPLIACION" -> "ampliacion_plazo": coincidencia directa (ampliación del plazo).
# - "PRORROGA" -> "ampliacion_plazo": una prórroga es, en esencia, una extensión/
#   diferimiento del plazo de pago; es el tipo del catálogo más cercano.
# - "CONSOLIDACION" -> "reestructuracion": consolidar obligaciones es una forma de
#   reestructuración de la deuda.
# - "COMBO", "CONDONACION", "PERIODO DE GRACIA", "OTROS": deliberadamente SIN mapear.
#   "COMBO" combina medidas no identificables individualmente; "CONDONACION" es una
#   condonación/perdón de deuda sin análogo en el catálogo; "PERIODO DE GRACIA" es una
#   suspensión temporal de pago, no una reducción de cuota ni una reestructuración;
#   "OTROS" es demasiado ambiguo (y tiene solo 3 filas en todo el dataset real). Mapear
#   cualquiera de estos con baja confianza sería adivinar, no derivar: se deja sin
#   mapear y por tanto no genera entrada en `ofertas_aplicadas_mes`.
_ALTERNATIVA_AGR_A_TIPO_OPCION: dict[str, str] = {
    "AMPLIACION": "ampliacion_plazo",
    "PRORROGA": "ampliacion_plazo",
    "CONSOLIDACION": "reestructuracion",
}

# Mismos valores por defecto que `config/reglas_elegibilidad.yaml` (no se importa ese
# archivo para no acoplar este provider a la ruta de configuración del
# `EligibilityEngine`; se hardcodea aquí y se mantiene consistente a propósito).
_COOLDOWN_MESES_POR_TIPO: dict[str, int] = {
    "ampliacion_plazo": 3,
    "reduccion_cuota": 3,
    "renegociacion_tasa": 4,
    "reestructuracion": 4,
}


class RawDataUnavailableError(Exception):
    """El CSV crudo requerido no existe o no pudo leerse (fail-fast, en el constructor)."""

    def __init__(self, path: Path, causa: Exception | None = None) -> None:
        self.path = path
        detalle = f" ({causa!r})" if causa is not None else ""
        super().__init__(f"No se pudo cargar el archivo de datos crudos '{path}'{detalle}")


def _build_id_obligacion(df: pd.DataFrame) -> pd.Series:
    """Reimplementación exacta de `ingest._build_id_obligacion` (sin import cruzado)."""
    return (
        df["nit_enmascarado"].astype(str)
        + "#"
        + df["num_oblig_orig_enmascarado"].astype(str)
        + "#"
        + df["num_oblig_enmascarado"].astype(str)
    )


def _fecha_aplicacion_desde_yyyymm(valor: Any) -> str:
    """Convierte `fecha_var_rpta_alt` (entero `YYYYMM`) al primer día ISO de ese mes."""
    texto = str(int(valor))
    return f"{texto[:4]}-{texto[4:6]}-01"


class CsvObligacionProfileProvider:
    """`ObligacionProfileProvider` real, respaldado por el CSV `trtest` de `modelo-propension`.

    Adaptador de datos reales best-effort (ver docstring del módulo para el detalle de
    cada decisión de mapeo y la brecha de cobertura conocida: solo ~49% de las
    obligaciones reales tienen historial en `trtest`; para el resto, `get_profile` lanza
    `ObligacionProfileNotFoundError`, que `ContextBuilder` ya sabe tratar como "perfil
    ausente" en lugar de fabricar datos).

    Cumple estructuralmente el Protocol `ObligacionProfileProvider` de `context_builder.py`
    (duck typing, igual que `InMemoryObligacionProfileProvider`).
    """

    def __init__(self, raw_dir: str | Path) -> None:
        """Carga y indexa (eager, no perezoso) el CSV `trtest` de `raw_dir`.

        Args:
            raw_dir: directorio que contiene los CSV crudos de `modelo-propension`
                (ver `RAW_DATA_DIR` en `settings.py`).

        Raises:
            RawDataUnavailableError: si el archivo no existe o no puede leerse (fail-fast:
                nunca se crea un provider "vacío" que falle silenciosamente después).
        """
        self._csv_path = Path(raw_dir) / TRTEST_FILENAME
        self._index: dict[str, dict[str, Any]] = self._load_index(self._csv_path)

    @staticmethod
    def _load_index(csv_path: Path) -> dict[str, dict[str, Any]]:
        if not csv_path.is_file():
            raise RawDataUnavailableError(csv_path)
        try:
            df = pd.read_csv(csv_path, usecols=_USECOLS)
        except Exception as exc:  # noqa: BLE001 - fail-fast: cualquier error de lectura es fatal aquí
            raise RawDataUnavailableError(csv_path, exc) from exc

        df = df.assign(id_obligacion=_build_id_obligacion(df))
        # Fila más reciente por id_obligacion: ordenar ascendente por fecha y quedarse con
        # la última aparición de cada id (equivalente a max(fecha_var_rpta_alt) por grupo,
        # pero evita un groupby+idxmax adicional sobre 500k+ filas).
        df_ordenado = df.sort_values("fecha_var_rpta_alt")
        mas_reciente = df_ordenado.drop_duplicates(subset="id_obligacion", keep="last")
        return mas_reciente.set_index("id_obligacion").to_dict(orient="index")

    def get_profile(self, id_obligacion: str) -> ObligacionProfileData:
        """Retorna el perfil derivado de la fila más reciente de `trtest` para `id_obligacion`.

        Raises:
            ObligacionProfileNotFoundError: si `id_obligacion` no tiene ninguna fila en
                `trtest` (obligación sin historial real conocido — ver brecha de
                cobertura documentada en el docstring del módulo).
        """
        fila = self._index.get(id_obligacion)
        if fila is None:
            raise ObligacionProfileNotFoundError(id_obligacion)
        return self._map_row_to_profile(fila)

    @staticmethod
    def _map_row_to_profile(fila: dict[str, Any]) -> ObligacionProfileData:
        """Aplica el mapeo de campos documentado en el docstring del módulo a una fila cruda."""
        dias_mora_fin = fila.get("dias_mora_fin")
        dias_mora = int(dias_mora_fin) if pd.notna(dias_mora_fin) else None

        vlr_vencido = fila.get("vlr_vencido")
        vlr_obligacion = fila.get("vlr_obligacion")
        if pd.notna(vlr_vencido):
            exposicion: float | None = float(vlr_vencido)
        elif pd.notna(vlr_obligacion):
            exposicion = float(vlr_obligacion)
        else:
            exposicion = None

        segmento_valor = fila.get("segmento")
        segmento = str(segmento_valor) if pd.notna(segmento_valor) else None

        canal_gestion_valor = fila.get("aplicativo")
        canal_gestion = str(canal_gestion_valor) if pd.notna(canal_gestion_valor) else None

        nit_valor = fila.get("nit_enmascarado")
        id_cliente = str(nit_valor) if pd.notna(nit_valor) else None

        ofertas_aplicadas_mes: list[OfertaAplicada] = []
        marca_alt_apli = fila.get("marca_alt_apli")
        alternativa = fila.get("alternativa_aplicada_agr")
        if marca_alt_apli == "SI" and pd.notna(alternativa):
            tipo_opcion = _ALTERNATIVA_AGR_A_TIPO_OPCION.get(str(alternativa).strip())
            if tipo_opcion is not None:
                ofertas_aplicadas_mes = [
                    OfertaAplicada(
                        tipo_opcion=tipo_opcion,
                        fecha_aplicacion=_fecha_aplicacion_desde_yyyymm(fila.get("fecha_var_rpta_alt")),
                        cooldown_meses=_COOLDOWN_MESES_POR_TIPO[tipo_opcion],
                    )
                ]

        ultima_opcion_aplicada = ofertas_aplicadas_mes[0] if ofertas_aplicadas_mes else None

        cant_acuerdo_binario = fila.get("cant_acuerdo_binario")
        # Doble señal exigida a propósito (ver docstring del módulo): evita marcar
        # acuerdo_pago_vigente=True sin ultima_opcion_aplicada que lo sustente, lo que
        # ContextBuilder ya trata como contradicción (Requirement 3.5).
        acuerdo_pago_vigente = bool(
            pd.notna(cant_acuerdo_binario)
            and int(cant_acuerdo_binario) == 1
            and ultima_opcion_aplicada is not None
        )

        return ObligacionProfileData(
            id_cliente=id_cliente,
            dias_mora=dias_mora,
            exposicion=exposicion,
            segmento=segmento,
            canal_gestion=canal_gestion,
            ofertas_aplicadas_mes=ofertas_aplicadas_mes,
            ultima_opcion_aplicada=ultima_opcion_aplicada,
            acuerdo_pago_vigente=acuerdo_pago_vigente,
            restriccion_vigente=False,  # brecha de datos conocida, ver docstring del módulo
        )
