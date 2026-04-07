# Arquitectura del Proyecto

## Principios

- Separacion clara entre codigo fuente, datos, configuracion y artefactos.
- Paquete Python unico bajo `src/tesis_ia`.
- Configuracion desacoplada para facilitar experimentos reproducibles.
- Estructura pensada para tesis: entendible, mantenible y extensible.

## Dominios principales

- `data/`: ingesta, validacion, transformacion y particionado de datos.
- `models/`: definicion de arquitecturas y sus componentes.
- `training/`: orquestacion del entrenamiento y checkpoints.
- `evaluation/`: metricas, analisis y reportes.
- `utils/`: utilidades compartidas y realmente transversales.

## Capas externas al codigo

- `config/`: parametros base, por dominio y por experimento.
- `data/`: datasets y splits persistidos.
- `artifacts/`: modelos entrenados, checkpoints y metadata.
- `outputs/`: reportes, figuras y predicciones.
- `notebooks/`: exploracion y prototipado.
