"""Implementación del Registro de Trazabilidad (`TraceLogger`).

Ver Component 8 de `design.md` y Requirement 10 de `requirements.md`.

Provee:
- `TraceLogger`: interfaz abstracta (ABC) que define el contrato append-only usado por el
  `Orchestrator` (Requirements 10.1, 10.2, 10.3, 10.5).
- `JsonlTraceLogger`: implementación concreta con almacenamiento en un archivo JSONL
  (ver `design.md`, sección "Example Usage": `JsonlTraceLogger(path="data_output/trace_log.jsonl")`).
- `DuplicateTraceIdError`: excepción lanzada al intentar reescribir un `trace_id` ya existente
  (rechazo explícito de update/delete, Requirement 10.3).
- `TraceNotFoundError`: excepción lanzada al consultar un `trace_id` inexistente.
"""
from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sistema_agentico.types import (
    EligibilityResult,
    EscalationReason,
    InputMode,
    OfertaElegible,
    ScoreInfo,
    TipoOferta,
    TraceRecord,
    ValidationResult,
    WhitelistItem,
)

DEFAULT_TRACE_LOG_PATH = "data_output/trace_log.jsonl"


class DuplicateTraceIdError(ValueError):
    """Se lanza al intentar `append` un `TraceRecord` cuyo `trace_id` ya fue escrito.

    El `TraceLogger` es append-only (RNF-05, Requirement 10.3): no existen operaciones de
    actualización ni eliminación sobre registros ya escritos; cualquier intento de reescribir
    un `trace_id` existente se rechaza explícitamente lanzando esta excepción.
    """


class TraceNotFoundError(KeyError):
    """Se lanza al consultar un `trace_id` que no existe en el log."""


class TraceLogger(ABC):
    """Contrato del Registro de Trazabilidad (Component 8 de `design.md`).

    Append-only por diseño: ninguna implementación debe exponer métodos de actualización o
    eliminación sobre registros ya escritos (Requirement 10.3). Las implementaciones concretas
    (p.ej. `JsonlTraceLogger`) solo pueden añadir (`append`) y consultar (`get_trace`,
    `get_history`) registros.
    """

    @abstractmethod
    def append(self, record: TraceRecord) -> None:
        """Escritura append-only; nunca se actualiza ni borra un registro existente.

        Raises:
            DuplicateTraceIdError: si ``record.trace_id`` ya fue escrito previamente.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trace(self, trace_id: str) -> TraceRecord:
        """Retorna el `TraceRecord` asociado a `trace_id`.

        Raises:
            TraceNotFoundError: si no existe ningún registro con ese `trace_id`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_history(self, id_obligacion: str) -> list[TraceRecord]:
        """Retorna todos los `TraceRecord` asociados a `id_obligacion`, sin omisiones ni duplicados."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Serialización JSON de `TraceRecord` (dataclasses/enums anidados -> dict -> dataclasses)
# ---------------------------------------------------------------------------


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _oferta_elegible_to_dict(oferta: OfertaElegible) -> dict[str, Any]:
    return {
        "tipo": _enum_value(oferta.tipo),
        "id_opcion": oferta.id_opcion,
        "detalle": oferta.detalle,
        "razon_elegibilidad": oferta.razon_elegibilidad,
    }


def _oferta_elegible_from_dict(data: dict[str, Any]) -> OfertaElegible:
    return OfertaElegible(
        tipo=TipoOferta(data["tipo"]),
        id_opcion=data["id_opcion"],
        detalle=data["detalle"],
        razon_elegibilidad=data["razon_elegibilidad"],
    )


def _score_info_to_dict(score_info: ScoreInfo) -> dict[str, Any]:
    return asdict(score_info)


def _score_info_from_dict(data: dict[str, Any]) -> ScoreInfo:
    return ScoreInfo(**data)


def _eligibility_result_to_dict(result: EligibilityResult) -> dict[str, Any]:
    return {
        "opciones_elegibles": [_oferta_elegible_to_dict(o) for o in result.opciones_elegibles],
        "acuerdo_elegible": (
            _oferta_elegible_to_dict(result.acuerdo_elegible) if result.acuerdo_elegible else None
        ),
        "motivo_no_elegible": result.motivo_no_elegible,
    }


def _eligibility_result_from_dict(data: dict[str, Any]) -> EligibilityResult:
    return EligibilityResult(
        opciones_elegibles=[_oferta_elegible_from_dict(o) for o in data["opciones_elegibles"]],
        acuerdo_elegible=(
            _oferta_elegible_from_dict(data["acuerdo_elegible"]) if data["acuerdo_elegible"] else None
        ),
        motivo_no_elegible=data["motivo_no_elegible"],
    )


def _whitelist_item_to_dict(item: WhitelistItem) -> dict[str, Any]:
    return {
        "oferta": _oferta_elegible_to_dict(item.oferta),
        "prioridad": item.prioridad,
        "justificacion": item.justificacion,
    }


def _whitelist_item_from_dict(data: dict[str, Any]) -> WhitelistItem:
    return WhitelistItem(
        oferta=_oferta_elegible_from_dict(data["oferta"]),
        prioridad=data["prioridad"],
        justificacion=data["justificacion"],
    )


def _validation_result_to_dict(result: ValidationResult) -> dict[str, Any]:
    return {
        "valido": result.valido,
        "motivo_rechazo": result.motivo_rechazo,
        "escalar": result.escalar,
        "razon_escalamiento": (
            _enum_value(result.razon_escalamiento) if result.razon_escalamiento else None
        ),
    }


def _validation_result_from_dict(data: dict[str, Any]) -> ValidationResult:
    return ValidationResult(
        valido=data["valido"],
        motivo_rechazo=data["motivo_rechazo"],
        escalar=data["escalar"],
        razon_escalamiento=(
            EscalationReason(data["razon_escalamiento"]) if data["razon_escalamiento"] else None
        ),
    )


def trace_record_to_dict(record: TraceRecord) -> dict[str, Any]:
    """Serializa un `TraceRecord` (incluyendo dataclasses/enums anidados) a un `dict` JSON-serializable."""
    return {
        "trace_id": record.trace_id,
        "timestamp": record.timestamp,
        "id_obligacion": record.id_obligacion,
        "modo": _enum_value(record.modo),
        "score_info": _score_info_to_dict(record.score_info),
        "reglas_evaluadas": record.reglas_evaluadas,
        "eligibility_result": _eligibility_result_to_dict(record.eligibility_result),
        "whitelist": [_whitelist_item_to_dict(w) for w in record.whitelist],
        "ofertas_mencionadas": list(record.ofertas_mencionadas),
        "respuesta_agente": record.respuesta_agente,
        "agente_responsable": record.agente_responsable,
        "validation_result": _validation_result_to_dict(record.validation_result),
        "prompt_version": record.prompt_version,
        "version_modelo_propension": record.version_modelo_propension,
    }


def trace_record_from_dict(data: dict[str, Any]) -> TraceRecord:
    """Reconstruye un `TraceRecord` completo a partir de su representación `dict`/JSON."""
    return TraceRecord(
        trace_id=data["trace_id"],
        timestamp=data["timestamp"],
        id_obligacion=data["id_obligacion"],
        modo=InputMode(data["modo"]),
        score_info=_score_info_from_dict(data["score_info"]),
        reglas_evaluadas=data["reglas_evaluadas"],
        eligibility_result=_eligibility_result_from_dict(data["eligibility_result"]),
        whitelist=[_whitelist_item_from_dict(w) for w in data["whitelist"]],
        ofertas_mencionadas=list(data["ofertas_mencionadas"]),
        respuesta_agente=data["respuesta_agente"],
        agente_responsable=data["agente_responsable"],
        validation_result=_validation_result_from_dict(data["validation_result"]),
        prompt_version=data["prompt_version"],
        version_modelo_propension=data["version_modelo_propension"],
    )


class JsonlTraceLogger(TraceLogger):
    """`TraceLogger` con almacenamiento append-only en un archivo JSONL (RF-21, RNF-05).

    Cada línea del archivo es un objeto JSON correspondiente a un `TraceRecord` serializado
    mediante `trace_record_to_dict`. La unicidad de `trace_id` (Requirement 10.2) y el rechazo
    de actualizaciones/eliminaciones sobre un `trace_id` ya escrito (Requirement 10.3) se
    garantizan manteniendo un índice en memoria (`trace_id -> TraceRecord`) que se reconstruye
    a partir del archivo en cada instancia y se actualiza en cada `append` exitoso.

    No expone ningún método de actualización o eliminación: la única forma de escribir es
    `append`, y este rechaza explícitamente cualquier `trace_id` ya presente en el log.
    """

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_TRACE_LOG_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        self._index: dict[str, TraceRecord] = {}
        self._load_index()

    @property
    def path(self) -> Path:
        """Ruta del archivo JSONL de almacenamiento."""
        return self._path

    def _load_index(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = trace_record_from_dict(json.loads(line))
                self._index[record.trace_id] = record

    def append(self, record: TraceRecord) -> None:
        """Escritura append-only; nunca se actualiza ni borra un registro existente.

        Raises:
            DuplicateTraceIdError: si ``record.trace_id`` ya fue escrito previamente
                (Requirement 10.3: rechazo explícito de update/delete sobre un trace_id existente).
        """
        with self._lock:
            if record.trace_id in self._index:
                raise DuplicateTraceIdError(
                    f"El trace_id '{record.trace_id}' ya fue escrito; TraceLogger es "
                    "append-only y rechaza actualizaciones/eliminaciones (Requirement 10.3)."
                )
            line = json.dumps(trace_record_to_dict(record), ensure_ascii=False)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._index[record.trace_id] = record

    def get_trace(self, trace_id: str) -> TraceRecord:
        """Retorna el `TraceRecord` asociado a `trace_id`.

        Raises:
            TraceNotFoundError: si no existe ningún registro con ese `trace_id`.
        """
        record = self._index.get(trace_id)
        if record is None:
            raise TraceNotFoundError(f"No existe ningún TraceRecord con trace_id='{trace_id}'")
        return record

    def get_history(self, id_obligacion: str) -> list[TraceRecord]:
        """Retorna todos los `TraceRecord` asociados a `id_obligacion`, en orden de escritura."""
        return [record for record in self._index.values() if record.id_obligacion == id_obligacion]
