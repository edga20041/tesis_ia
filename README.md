# tesis_ia — Microservicio de Inteligencia Artificial

Microservicio de **pronóstico de demanda** de la Plataforma de Inteligencia Predictiva B2B (Proyecto Final de Ingeniería, UADE). Se expone como API REST con FastAPI y lo consume el backend transaccional (`tesis_back`), que a su vez sirve al frontend (`tesis_front`).

## Qué hace

| Capacidad | Descripción | Módulo |
|---|---|---|
| **Cierre mensual** | Sincroniza variables exógenas, recalcula *features*, entrena el modelo y persiste predicciones a 12 meses | `training/db_trainer_predictor.py`, `data/` |
| **Modelo Champion** | Random Forest sobre `log1p(unidades)` más un especialista de razón acotado por un techo relativo (K = 1,25): modelo híbrido | `training/db_trainer_predictor.py` |
| **Simulador *what-if*** | Proyecta 12 meses variando precio, tipo de cambio e IPC | `training/db_trainer_predictor.py` |
| **Arranque en frío** | Estima la demanda de un producto sin historial por similitud con productos maduros (K-Means, KNN, TF-IDF, rampa de Bass, elasticidad de precio) | `models/cold_start.py` |
| **Explicabilidad** | Importancia de variables del modelo vigente | `training/db_trainer_predictor.py` |
| **Intervalos de confianza** | Cuantiles P10, P50 y P90 a partir de la varianza del ensamble | `training/db_trainer_predictor.py` |

El modelo vigente en producción es `champion_rf_2025_12`, con **MAE 209,22, RMSE 2.509,10 y MAPE 15,66 %** (validación fuera de muestra).

## API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servicio y de la conexión a la base |
| `POST` | `/pipeline/run-cierre-mensual` | Ejecuta el cierre mensual de un período (`anio`, `mes`) |
| `POST` | `/simulation/predict` | Simulación de escenarios de un producto |
| `POST` | `/cold-start/predict` | Proyección de un producto nuevo sin historial |
| `GET` | `/model/explainability` | Importancia de variables del modelo |

Contratos detallados en [`docs/api.md`](docs/api.md). La documentación interactiva (Swagger) está en `/docs` con el servicio en ejecución.

## Estructura del código

```text
src/tesis_ia/
├── api.py                     # API FastAPI (punto de entrada del servicio)
├── config.py                  # Carga y fusión de la configuración YAML
├── data/
│   ├── exogenous_pipeline.py  # Clima (Open-Meteo), macro (IMF / Banco Mundial), feriados
│   └── feature_calculator.py  # Consolidación semanal y features (lags, medias móviles, cuota de mercado)
├── models/
│   ├── factory.py             # Familias: regresión lineal, árbol, Random Forest, LightGBM, XGBoost
│   └── cold_start.py          # Motor de arranque en frío
├── training/
│   └── db_trainer_predictor.py  # Entrenamiento, predicción, simulación y explicabilidad sobre la base
└── utils/
    └── db_connector.py        # Conexión a MySQL (SQLAlchemy)
scripts/
├── cron_cierre_mensual.sh     # Dispara el cierre mensual (crontab del VPS)
├── experiment_tournament.py   # Torneo de modelos (resultados en la bitácora, fase 4)
└── ...                        # Scripts de experimentación
tests/                         # 165 pruebas (pytest)
```

> **Nota.** `cli.py`, `pipeline.py`, `training/trainer.py`, `evaluation/` y los *scripts* `train.py`, `predict.py`, `evaluate.py` y `download_data.py` provienen de la plantilla original del proyecto. **El servicio en producción no los utiliza** (el contenedor ejecuta únicamente `tesis_ia.api:app`). Sus puntos de entrada devuelven un estado `planned`.

## Configuración

Variables de entorno de la conexión a la base (`utils/db_connector.py`):

| Variable | Valor por defecto |
|---|---|
| `DB_HOST` | `localhost` |
| `DB_PORT` | `3306` |
| `DB_NAME` | `tesis` |
| `DB_USER` | `root` |
| `DB_PASSWORD` | (vacío) |

La configuración YAML de `config/` (`config.yaml` unificado y archivos por dominio) la usan los scripts de entrenamiento y el torneo. En `config/model.yaml` figura una arquitectura de red neuronal heredada de la plantilla que **no** se entrena; el modelo real se selecciona en el código (`model_family="random_forest"`).

## Ejecución

### Con Docker (producción)

```bash
docker compose up --build      # expone el puerto 8000
```

La imagen usa **Python 3.10** (`Dockerfile`) y arranca `uvicorn tesis_ia.api:app`. En el VPS, el contenedor se despliega mediante Portainer y el backend lo consume por la red interna `tesis-network` (`http://tesis-ia-api:8000`).

> **Requisito operativo.** El servicio precarga el modelo al arrancar (`startup_event`), por lo que tarda varios minutos en responder tras un reinicio. Durante el cierre mensual el consumo de memoria alcanza un pico cercano a 2,3 GiB: el contenedor debe ejecutarse con un límite explícito (`--memory=3g`).

### Local

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src uvicorn tesis_ia.api:app --port 8000
```

Las versiones fijadas en `requirements.txt` no compilan en intérpretes posteriores a Python 3.10.

## Pruebas

```bash
pip install pytest==7.4.0 pytest-cov==4.1.0 httpx
python -m pytest --cov=src/tesis_ia
```

Son 165 pruebas: lógica de arranque en frío, modelo híbrido, ausencia de fuga de datos en las *features*, configuración y capa REST (con dobles para la base y el modelo). El flujo de GitHub Actions (`.github/workflows/tests.yml`) las ejecuta en cada *push* y *pull request* hacia `master`. No están cubiertos los métodos que dependen de la base (`train_and_evaluate`, `generate_and_save_predictions`, etc.).

## Documentación

- [`docs/api.md`](docs/api.md): contratos de la API.
- [`docs/architecture.md`](docs/architecture.md) y [`docs/troubleshooting.md`](docs/troubleshooting.md): documentos genéricos de la plantilla.

## Autores

Edgardo Crespo y Felipe Calderón.
