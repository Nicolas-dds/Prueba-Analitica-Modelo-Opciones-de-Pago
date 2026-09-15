"""Generación de datos sintéticos de clientes, obligaciones e historial (RF-23).

- `profiles`: perfiles sintéticos de cliente/obligación (`ClienteObligacionContext`),
  incluyendo historial de ofertas aplicadas (`OfertaAplicada`) — tarea 4.1.
- `conversations`: mensajes reactivos de cliente (`SyntheticConversation`) alineados a
  los 7 escenarios mínimos del reto, combinables con `profiles` para armar `RawInput`
  de prueba — tarea 4.2.
"""
