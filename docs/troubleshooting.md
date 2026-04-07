# Troubleshooting

## Problemas comunes

### ImportError del paquete
- Verificar que el paquete viva en `src/tesis_ia`.
- Instalar el proyecto en modo editable con `pip install -e .`.

### Configuraciones mezcladas o duplicadas
- Usar `config/config.yaml` para pruebas rapidas.
- Usar los archivos separados y `config/experiments/` para experimentos reales.

### Artefactos pesados dentro de git
- Guardar modelos y salidas en `artifacts/` y `outputs/`.
- Confirmar que `.gitignore` siga alineado con esas carpetas.
