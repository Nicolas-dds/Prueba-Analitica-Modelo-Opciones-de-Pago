"""Deterministic payment-option eligibility evaluation (Component 4).

``EligibilityEngine`` has no client dependencies and does not import conversational or
network components.  It evaluates only the stable, pre-approved catalogue below.  For
cooldowns, the latest matching item among ``ofertas_aplicadas_mes`` and
``ultima_opcion_aplicada`` is authoritative; this prevents a recent monthly application
from being missed when the legacy ``ultima_opcion_aplicada`` field is stale.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date

from sistema_agentico.eligibility.reglas import ReglasElegibilidad
from sistema_agentico.types import (
    ClienteObligacionContext,
    EligibilityResult,
    OfertaAplicada,
    OfertaElegible,
    TipoOferta,
)

# The result contract in ``types.py`` and RF-15 impose this non-configurable ceiling.
MAX_OPCIONES_ELEGIBLES = 3


@dataclass(frozen=True)
class _OpcionPreaprobada:
    """An immutable catalogue item; its order is the deterministic evaluation order."""

    tipo_opcion: str
    id_opcion: str
    detalle: dict[str, str | bool]


# Stable pre-approved catalogue.  IDs intentionally match their business option type,
# which makes downstream whitelist/audit references durable and unambiguous.
_CATALOGO_OPCIONES_PREAPROBADAS: tuple[_OpcionPreaprobada, ...] = (
    _OpcionPreaprobada(
        tipo_opcion="ampliacion_plazo",
        id_opcion="ampliacion_plazo",
        detalle={"tipo_opcion": "ampliacion_plazo", "preaprobada": True},
    ),
    _OpcionPreaprobada(
        tipo_opcion="reduccion_cuota",
        id_opcion="reduccion_cuota",
        detalle={"tipo_opcion": "reduccion_cuota", "preaprobada": True},
    ),
    _OpcionPreaprobada(
        tipo_opcion="renegociacion_tasa",
        id_opcion="renegociacion_tasa",
        detalle={"tipo_opcion": "renegociacion_tasa", "preaprobada": True},
    ),
    _OpcionPreaprobada(
        tipo_opcion="reestructuracion",
        id_opcion="reestructuracion",
        detalle={"tipo_opcion": "reestructuracion", "preaprobada": True},
    ),
)


class EligibilityEngine:
    """Evaluate eligibility from injected rules and an explicit deterministic date.

    Callers must either pass ``fecha_actual`` to :meth:`evaluate` or inject ``clock``.
    This deliberately avoids reading wall-clock time implicitly, so a fixed context,
    rules version, and date always yield the same result.
    """

    def __init__(
        self,
        reglas: ReglasElegibilidad,
        *,
        clock: Callable[[], date] | None = None,
    ) -> None:
        self._reglas = reglas
        self._clock = clock

    def evaluate(
        self,
        contexto: ClienteObligacionContext,
        *,
        fecha_actual: date | None = None,
        incumplimiento_acuerdo_previo: bool = False,
    ) -> EligibilityResult:
        """Return only pre-approved options and, when allowed, a payment agreement.

        ``incumplimiento_acuerdo_previo`` is an explicit one-interaction signal from
        ``ContextBuilder``.  When true, it suppresses only this call's payment
        agreement; normal payment options remain independently evaluable.  The engine
        does not retain or mutate the signal, so a subsequent call without it follows
        the normal agreement rules.

        The configured monthly limit is honored up to the immutable RF-15 result
        ceiling of three options.  Cooldowns use complete calendar months: an option
        applied on the 15th is eligible again on the 15th after its configured number
        of months, never by an approximation of thirty-day periods.
        """
        evaluacion_fecha = self._resolve_fecha_actual(fecha_actual)
        limite_opciones = min(
            self._reglas.max_opciones_pago_mes, MAX_OPCIONES_ELEGIBLES
        )
        opciones_elegibles: list[OfertaElegible] = []

        if len(contexto.ofertas_aplicadas_mes) < limite_opciones:
            for opcion in _CATALOGO_OPCIONES_PREAPROBADAS:
                if len(opciones_elegibles) >= limite_opciones:
                    break
                ultima_aplicada = self._ultima_aplicada_del_tipo(
                    contexto, opcion.tipo_opcion
                )
                if self._en_cooldown(
                    ultima_aplicada, opcion.tipo_opcion, evaluacion_fecha
                ):
                    continue
                opciones_elegibles.append(
                    OfertaElegible(
                        tipo=TipoOferta.OPCION_PAGO,
                        id_opcion=opcion.id_opcion,
                        detalle=dict(opcion.detalle),
                        razon_elegibilidad=(
                            "Opción preaprobada sin cooldown activo para "
                            f"{opcion.tipo_opcion}."
                        ),
                    )
                )

        acuerdo_elegible = self._evaluar_acuerdo(
            contexto, incumplimiento_acuerdo_previo=incumplimiento_acuerdo_previo
        )
        motivo_no_elegible = None
        if not opciones_elegibles and acuerdo_elegible is None:
            motivo_no_elegible = self._explicar_no_elegibilidad(
                contexto,
                limite_opciones,
                evaluacion_fecha,
                incumplimiento_acuerdo_previo=incumplimiento_acuerdo_previo,
            )

        return EligibilityResult(
            opciones_elegibles=opciones_elegibles,
            acuerdo_elegible=acuerdo_elegible,
            motivo_no_elegible=motivo_no_elegible,
        )

    def _resolve_fecha_actual(self, fecha_actual: date | None) -> date:
        if fecha_actual is not None:
            return fecha_actual
        if self._clock is None:
            raise ValueError(
                "fecha_actual es obligatoria cuando EligibilityEngine no recibe un clock"
            )
        resolved = self._clock()
        if not isinstance(resolved, date):
            raise TypeError("clock debe retornar datetime.date")
        return resolved

    def _evaluar_acuerdo(
        self,
        contexto: ClienteObligacionContext,
        *,
        incumplimiento_acuerdo_previo: bool,
    ) -> OfertaElegible | None:
        if (
            incumplimiento_acuerdo_previo
            or contexto.acuerdo_pago_vigente
            or contexto.restriccion_vigente
        ):
            return None
        return OfertaElegible(
            tipo=TipoOferta.ACUERDO_PAGO,
            id_opcion="acuerdo_pago_estandar",
            detalle={"max_dias_compromiso": self._reglas.max_dias_acuerdo},
            razon_elegibilidad=(
                "Cliente sin acuerdo de pago vigente ni restricción vigente."
            ),
        )

    def _ultima_aplicada_del_tipo(
        self, contexto: ClienteObligacionContext, tipo_opcion: str
    ) -> OfertaAplicada | None:
        """Find the latest known application for a type across both context fields."""
        candidatas = (
            oferta
            for oferta in self._historial_con_ultima(contexto)
            if oferta.tipo_opcion == tipo_opcion
        )
        return max(candidatas, key=self._fecha_de_aplicacion, default=None)

    @staticmethod
    def _historial_con_ultima(
        contexto: ClienteObligacionContext,
    ) -> Iterable[OfertaAplicada]:
        yield from contexto.ofertas_aplicadas_mes
        if contexto.ultima_opcion_aplicada is not None:
            yield contexto.ultima_opcion_aplicada

    def _en_cooldown(
        self,
        ultima_aplicada: OfertaAplicada | None,
        tipo_opcion: str,
        fecha_actual: date,
    ) -> bool:
        if ultima_aplicada is None:
            return False
        meses_transcurridos = self._meses_calendario_completos(
            self._fecha_de_aplicacion(ultima_aplicada), fecha_actual
        )
        return meses_transcurridos < self._reglas.cooldown_meses_por_tipo[tipo_opcion]

    @staticmethod
    def _fecha_de_aplicacion(oferta: OfertaAplicada) -> date:
        try:
            return date.fromisoformat(oferta.fecha_aplicacion)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "fecha_aplicacion debe ser una fecha ISO válida (YYYY-MM-DD)"
            ) from error

    @staticmethod
    def _meses_calendario_completos(fecha_inicio: date, fecha_fin: date) -> int:
        """Return completed calendar months, preserving exact day-of-month boundaries."""
        meses = (fecha_fin.year - fecha_inicio.year) * 12 + (
            fecha_fin.month - fecha_inicio.month
        )
        if fecha_fin.day < fecha_inicio.day:
            meses -= 1
        return meses

    def _explicar_no_elegibilidad(
        self,
        contexto: ClienteObligacionContext,
        limite_opciones: int,
        fecha_actual: date,
        *,
        incumplimiento_acuerdo_previo: bool,
    ) -> str:
        motivos: list[str] = []
        if len(contexto.ofertas_aplicadas_mes) >= limite_opciones:
            motivos.append(
                "Se alcanzó el límite mensual de "
                f"{limite_opciones} opciones de pago aplicadas."
            )
        else:
            tipos_en_cooldown = [
                opcion.tipo_opcion
                for opcion in _CATALOGO_OPCIONES_PREAPROBADAS
                if self._en_cooldown(
                    self._ultima_aplicada_del_tipo(contexto, opcion.tipo_opcion),
                    opcion.tipo_opcion,
                    fecha_actual,
                )
            ]
            if tipos_en_cooldown:
                motivos.append(
                    "Cooldown activo para: " + ", ".join(tipos_en_cooldown) + "."
                )
            else:
                motivos.append("No hay opciones de pago preaprobadas disponibles.")
        if incumplimiento_acuerdo_previo:
            motivos.append(
                "El acuerdo de pago no está disponible en esta interacción posterior "
                "a su incumplimiento."
            )
        if contexto.acuerdo_pago_vigente:
            motivos.append("Existe un acuerdo de pago vigente.")
        if contexto.restriccion_vigente:
            motivos.append("Existe una restricción vigente sobre la obligación.")
        return " ".join(motivos)


__all__ = ["EligibilityEngine", "MAX_OPCIONES_ELEGIBLES"]
