"""Registro de Trazabilidad (`TraceLogger`).

Ver Component 8 de `design.md`.
"""
from sistema_agentico.trace.logger import (
    DEFAULT_TRACE_LOG_PATH,
    DuplicateTraceIdError,
    JsonlTraceLogger,
    TraceLogger,
    TraceNotFoundError,
    trace_record_from_dict,
    trace_record_to_dict,
)

__all__ = [
    "DEFAULT_TRACE_LOG_PATH",
    "DuplicateTraceIdError",
    "JsonlTraceLogger",
    "TraceLogger",
    "TraceNotFoundError",
    "trace_record_from_dict",
    "trace_record_to_dict",
]
