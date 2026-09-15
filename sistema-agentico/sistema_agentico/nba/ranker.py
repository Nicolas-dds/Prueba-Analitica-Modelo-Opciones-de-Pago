"""Deterministic Next Best Action ranking (Component 5).

The ranker is intentionally pure: it only reads ``EligibilityResult`` and
``ClienteObligacionContext`` and returns new immutable ``WhitelistItem`` records.
It does not call a model, network service, or mutate either input.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sistema_agentico.types import (
    ClienteObligacionContext,
    EligibilityResult,
    OfertaElegible,
    TipoOferta,
    WhitelistItem,
)


class NBARanker:
    """Create a deterministic, business-explainable whitelist of eligible offers.

    Each candidate receives a transparent score composed of: 50 points for payment
    propensity, 20 for exposure, 15 for days overdue, 15 for the offer's fit to
    the current situation, and a small history adjustment.  Propensity, exposure,
    and delinquency describe the obligation; offer fit and history distinguish the
    alternatives for that same obligation.

    Candidates are ordered by total score (descending), offer-fit score
    (descending), type order (payment options before an agreement), ``id_opcion``,
    and a canonical representation of the offer.  The last key makes duplicate-ID
    selection independent of the source-list order.  A payment agreement is a
    focused short-term alternative: it receives maximum fit only for a high-
    propensity, early-delinquency customer, and otherwise ranks below relief
    alternatives when their fit is higher.
    """

    _EXPOSICION_REFERENCIA = 1_000_000.0
    _MORA_REFERENCIA_DIAS = 180
    _TIPO_TIEBREAK = {
        TipoOferta.OPCION_PAGO: 0,
        TipoOferta.ACUERDO_PAGO: 1,
    }

    def rank(
        self,
        eligibility: EligibilityResult,
        contexto: ClienteObligacionContext,
    ) -> list[WhitelistItem]:
        """Return only received eligible offers with unique consecutive priorities.

        Duplicate candidate IDs are collapsed before ranking.  The canonical offer
        is selected using stable business fields, then every resulting item receives
        a newly assigned priority from one; priorities on any upstream object are
        never trusted or reused.
        """
        ofertas = self._ofertas_unicas(eligibility)
        ranked = sorted(
            ofertas,
            key=lambda oferta: self._orden_ranking(oferta, contexto),
        )
        return [
            WhitelistItem(
                oferta=oferta,
                prioridad=prioridad,
                justificacion=self._justificacion(oferta, contexto),
            )
            for prioridad, oferta in enumerate(ranked, start=1)
        ]

    @classmethod
    def _ofertas_unicas(
        cls, eligibility: EligibilityResult
    ) -> list[OfertaElegible]:
        """Select one canonical input offer per ID without modifying eligibility."""
        candidatas = list(eligibility.opciones_elegibles)
        if eligibility.acuerdo_elegible is not None:
            candidatas.append(eligibility.acuerdo_elegible)

        por_id: dict[str, OfertaElegible] = {}
        for oferta in candidatas:
            existente = por_id.get(oferta.id_opcion)
            if existente is None or cls._clave_canonica(oferta) < cls._clave_canonica(
                existente
            ):
                por_id[oferta.id_opcion] = oferta
        return list(por_id.values())

    def _orden_ranking(
        self,
        oferta: OfertaElegible,
        contexto: ClienteObligacionContext,
    ) -> tuple[float, float, int, str, tuple[Any, ...]]:
        ajuste_oferta = self._afinidad_oferta(oferta, contexto)
        total = self._puntaje_total(oferta, contexto, ajuste_oferta)
        return (
            -total,
            -ajuste_oferta,
            self._TIPO_TIEBREAK[oferta.tipo],
            oferta.id_opcion,
            self._clave_canonica(oferta),
        )

    def _puntaje_total(
        self,
        oferta: OfertaElegible,
        contexto: ClienteObligacionContext,
        afinidad_oferta: float,
    ) -> float:
        propension = contexto.score.score * 50
        exposicion = min(contexto.exposicion / self._EXPOSICION_REFERENCIA, 1.0) * 20
        mora = min(contexto.dias_mora / self._MORA_REFERENCIA_DIAS, 1.0) * 15
        historial = self._ajuste_historial(oferta, contexto)
        return round(propension + exposicion + mora + afinidad_oferta + historial, 6)

    def _afinidad_oferta(
        self,
        oferta: OfertaElegible,
        contexto: ClienteObligacionContext,
    ) -> float:
        """Return the deterministic 0..15 business fit for an eligible offer."""
        if oferta.tipo is TipoOferta.ACUERDO_PAGO:
            return (
                15.0
                if contexto.score.score >= 0.60 and contexto.dias_mora <= 30
                else 3.0
            )

        tipo_opcion = self._tipo_opcion(oferta)
        if tipo_opcion == "reduccion_cuota":
            return 15.0 if (contexto.exposicion >= 500_000 or contexto.dias_mora >= 60) else 9.0
        if tipo_opcion == "ampliacion_plazo":
            return 13.0 if contexto.dias_mora < 60 else 7.0
        if tipo_opcion == "renegociacion_tasa":
            return 12.0 if contexto.score.score >= 0.40 else 8.0
        if tipo_opcion == "reestructuracion":
            return 14.0 if (contexto.dias_mora >= 90 or contexto.exposicion >= 1_000_000) else 6.0
        return 5.0

    @staticmethod
    def _tipo_opcion(oferta: OfertaElegible) -> str:
        tipo = oferta.detalle.get("tipo_opcion", oferta.id_opcion)
        return tipo if isinstance(tipo, str) else oferta.id_opcion

    def _ajuste_historial(
        self,
        oferta: OfertaElegible,
        contexto: ClienteObligacionContext,
    ) -> float:
        """Avoid preferring a recently applied option while preserving eligibility."""
        tipo_opcion = self._tipo_opcion(oferta)
        if (
            contexto.ultima_opcion_aplicada is not None
            and contexto.ultima_opcion_aplicada.tipo_opcion == tipo_opcion
        ):
            return -5.0
        if any(
            aplicada.tipo_opcion == tipo_opcion
            for aplicada in contexto.ofertas_aplicadas_mes
        ):
            return -2.0
        return 0.0

    def _justificacion(
        self,
        oferta: OfertaElegible,
        contexto: ClienteObligacionContext,
    ) -> str:
        """Build a non-empty Spanish explanation using the complete ranking context."""
        historial = self._texto_historial(oferta, contexto)
        if oferta.tipo is TipoOferta.ACUERDO_PAGO:
            enfoque = (
                "El acuerdo de pago se prioriza como compromiso de corto plazo"
                if self._afinidad_oferta(oferta, contexto) == 15.0
                else "El acuerdo de pago se conserva como alternativa de compromiso de corto plazo"
            )
        else:
            enfoque = f"La opción {oferta.id_opcion} se ajusta a la necesidad de pago identificada"
        return (
            f"{enfoque}: score de propensión {contexto.score.score:.2f}, "
            f"exposición de {contexto.exposicion:.2f} y {contexto.dias_mora} días de mora. "
            f"{historial} Elegibilidad confirmada: {oferta.razon_elegibilidad}"
        )

    def _texto_historial(
        self,
        oferta: OfertaElegible,
        contexto: ClienteObligacionContext,
    ) -> str:
        tipo_opcion = self._tipo_opcion(oferta)
        if (
            contexto.ultima_opcion_aplicada is not None
            and contexto.ultima_opcion_aplicada.tipo_opcion == tipo_opcion
        ):
            return "El historial registra una aplicación previa del mismo tipo, por lo que se modera su prioridad."
        if contexto.ofertas_aplicadas_mes:
            return (
                "El historial incluye "
                f"{len(contexto.ofertas_aplicadas_mes)} oferta(s) aplicada(s) este mes; "
                "se prioriza una alternativa no repetida."
            )
        return "No hay ofertas aplicadas este mes que desplacen esta alternativa."

    @classmethod
    def _clave_canonica(cls, oferta: OfertaElegible) -> tuple[Any, ...]:
        return (
            oferta.tipo.value,
            oferta.id_opcion,
            cls._valor_estable(oferta.detalle),
            oferta.razon_elegibilidad,
        )

    @classmethod
    def _valor_estable(cls, valor: Any) -> tuple[Any, ...]:
        """Represent arbitrary detail data as a comparable, deterministic tuple."""
        if valor is None:
            return ("none",)
        if isinstance(valor, bool):
            return ("bool", valor)
        if isinstance(valor, (int, float, str)):
            return (type(valor).__name__, valor)
        if isinstance(valor, Mapping):
            return (
                "mapping",
                tuple(
                    sorted(
                        (str(clave), cls._valor_estable(dato))
                        for clave, dato in valor.items()
                    )
                ),
            )
        if isinstance(valor, (list, tuple)):
            return ("sequence", tuple(cls._valor_estable(dato) for dato in valor))
        return (type(valor).__module__, type(valor).__qualname__, str(valor))


__all__ = ["NBARanker"]
