"""Sistema Agéntico de Gestión de Cartera en Mora.

Paquete raíz del prototipo descrito en `.kiro/specs/sistema-agentico-cobranza/design.md`.
Cada submódulo corresponde a uno o más de los 10 componentes del diseño:

- `guardrails`: Guardrail de Entrada (Component 1) y Guardrail de Salida (Component 7).
- `orchestration`: Orquestador Determinista (Component 2).
- `context`: Context Builder (Component 3).
- `eligibility`: Motor de Elegibilidad (Component 4).
- `nba`: Ranking NBA (Component 5).
- `conversational`: Agente Conversacional (Component 6).
- `escalation`: Gestor de Escalamiento (parte de Component 7).
- `trace`: Registro de Trazabilidad (Component 8).
- `evaluation`: Dataset Dorado y Evaluación (Component 9).
- `human_review`: Gestor Humano y Auditor (Component 10).
- `synthetic_data`: Generación de datos sintéticos de prueba (RF-23).
"""
