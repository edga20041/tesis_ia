# API REST del microservicio de IA

Servicio FastAPI (`src/tesis_ia/api.py`), puerto 8000. Documentación interactiva generada automáticamente en `/docs` (Swagger) y esquema en `/openapi.json`.

Los errores no controlados de los *endpoints* de negocio se devuelven como `500` con el cuerpo `{"detail": "<mensaje>"}`; la validación del contrato responde `422`.

## `GET /health`

Estado del servicio y de la conexión a la base. Responde `200` incluso con la base caída, para que el *healthcheck* de Docker reciba el diagnóstico.

```json
{ "status": "healthy", "database_connected": true, "timestamp": "2026-09-25T17:00:00" }
```

`status` es `unhealthy` cuando la base no responde.

## `POST /pipeline/run-cierre-mensual`

Ejecuta el cierre mensual completo, en este orden: ingesta de variables exógenas → consolidación semanal y cálculo de *features* → entrenamiento del modelo Champion → persistencia de predicciones a 12 meses. Si la ingesta exógena falla, el cierre continúa con los valores previos. La versión del modelo se registra como `champion_rf_AAAA_MM`.

| Campo | Tipo | Restricción |
|---|---|---|
| `anio` | entero | 2000–2100 (obligatorio) |
| `mes` | entero | 1–12 (obligatorio) |
| `id_empresa` | texto | opcional |

Respuesta: `status`, `message`, `anio`, `mes`, `exogenas_sincronizadas`, `id_modelo`, `predicciones_generadas`.

## `POST /simulation/predict`

Simulación de escenarios a 12 meses para un producto, variando precio, tipo de cambio e IPC.

| Campo | Tipo | Por defecto |
|---|---|---|
| `id_producto` | texto (obligatorio) | — |
| `precio_pct` | decimal | `0.0` |
| `usd_pct` | decimal | `0.0` |
| `ipc_pct` | decimal | `0.0` |

Respuesta: `status`, `id_producto`, `variaciones_aplicadas` y `simulacion` (lista de meses proyectados).

## `POST /cold-start/predict`

Proyección de demanda de un producto nuevo, sin historial, por similitud con productos maduros del catálogo.

| Campo | Tipo | Restricción |
|---|---|---|
| `id_departamento` | entero | obligatorio |
| `id_categoria` | entero | obligatorio |
| `id_marca` | texto | obligatorio |
| `descripcion` | texto | por defecto `"Nuevo Producto"` |
| `contenido_neto` | decimal | obligatorio, mayor que 0 |
| `unidad_medida` | texto | obligatorio (`ml`, `lt`, `gr`, `kg`, `un`) |
| `valor_unitario` | decimal | obligatorio, mayor que 0 (guaraníes) |
| `horizonte_meses` | entero | 1–24, por defecto 12 |
| `anio_inicio`, `mes_inicio` | entero | opcionales (por defecto, el mes actual) |
| `id_empresa` | texto | opcional |

Respuesta: `status`, `producto_nuevo`, `productos_similares_gemelos` y `proyeccion_12_meses`.

## `GET /model/explainability`

Importancia de variables del modelo vigente. Campos: `status`, `modelo_tipo`, `total_variables`, `resumen_por_grupo` e `importancia_variables`. Los nombres viajan en *snake_case*.

## Integración con el backend

`tesis_back` consume estos *endpoints* con `RestClient` (`RealMlForecastClient`, `RealColdStartClient`), tomando la URL de la variable `ML_SERVICE_URL`. En producción debe apuntar al nombre interno del contenedor (`http://tesis-ia-api:8000`), no a `localhost`.

## Verificación

Este contrato está cubierto por `tests/test_api.py` (59 pruebas), que incluye la comprobación de que el documento OpenAPI expone exactamente estas cinco rutas.
