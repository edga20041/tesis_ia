# Proyecto de Tesis de IA

Base estructural para desarrollar, entrenar y evaluar modelos de inteligencia artificial con una organizacion clara, escalable y mantenible.

## Objetivo

Este repositorio deja preparado el esqueleto del proyecto para que la implementacion tecnica crezca sin mezclar responsabilidades entre codigo, datos, configuraciones, artefactos y resultados.

## Estructura Actual

```text
tesis_ia/
├── src/
│   └── tesis_ia/
│       ├── __init__.py
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       └── utils/
├── config/
│   ├── base.yaml
│   ├── config.yaml
│   ├── data.yaml
│   ├── model.yaml
│   ├── training.yaml
│   ├── evaluation.yaml
│   └── experiments/
│       └── baseline.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── artifacts/
│   ├── checkpoints/
│   ├── final/
│   └── metadata/
├── outputs/
│   ├── predictions/
│   ├── plots/
│   └── reports/
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── troubleshooting.md
├── notebooks/
├── scripts/
├── tests/
├── logs/
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── README.md
```

## Responsabilidades

| Carpeta | Responsabilidad |
|---|---|
| `src/tesis_ia/data` | Ingesta, validacion, preprocesamiento y particionado de datos |
| `src/tesis_ia/models` | Definicion de arquitecturas y componentes del modelo |
| `src/tesis_ia/training` | Orquestacion del entrenamiento, callbacks y checkpoints |
| `src/tesis_ia/evaluation` | Evaluacion, metricas, analisis y reportes |
| `src/tesis_ia/utils` | Utilidades realmente compartidas entre modulos |
| `config/` | Configuracion separada por dominio y por experimento |
| `data/` | Datos crudos, procesados y splits persistidos |
| `artifacts/` | Checkpoints, modelos finales y metadatos |
| `outputs/` | Predicciones, figuras y reportes generados |
| `notebooks/` | Exploracion y prototipado |
| `scripts/` | Puntos de entrada operativos del proyecto |
| `tests/` | Tests unitarios e integracion |

## Decisiones de Organizacion

- El paquete Python real vive en `src/tesis_ia`, no directamente en `src/`.
- Se usa `artifacts/` en lugar de una carpeta raiz `models/` para no confundir codigo con modelos entrenados.
- La configuracion se separa por dominio para facilitar experimentos reproducibles.
- `config/config.yaml` queda como archivo unificado para escenarios simples o pruebas rapidas.
- `utils/` queda reservado para helpers transversales; la logica de dominio debe vivir en su modulo correspondiente.

## Configuracion

### Config base

- `config/base.yaml`: parametros globales del proyecto y runtime.
- `config/data.yaml`: rutas y reglas de datos/preprocesamiento.
- `config/model.yaml`: definicion estructural del modelo y artefactos.
- `config/training.yaml`: hiperparametros de entrenamiento.
- `config/evaluation.yaml`: evaluacion e inferencia.
- `config/experiments/baseline.yaml`: stack sugerido para el experimento base.
- `config/config.yaml`: version consolidada para entornos simples.

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Uso Recomendado

### Desarrollo del paquete

```bash
python3 -m pip install -e .
```

### Verificacion de imports

```bash
PYTHONPATH=src python3 -c "import tesis_ia; print(tesis_ia.__version__)"
```

### Tests

```bash
pytest
pytest --cov=src/tesis_ia
```

## Estado del Proyecto

El repositorio esta preparado como esqueleto organizativo. Todavia no implementa la logica de entrenamiento, evaluacion o prediccion; la estructura quedo lista para empezar a construir esas capas sin reordenar todo despues.

## Documentacion

- Arquitectura: [docs/architecture.md](docs/architecture.md)
- API interna: [docs/api.md](docs/api.md)
- Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md)

## Autor

Edgardo Crespo y Felipe Calderon
