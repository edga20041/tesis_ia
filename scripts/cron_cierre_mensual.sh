#!/bin/bash
# ==============================================================================
# Script de automatización para Cierre Mensual del Microservicio de IA
# Ejecutado automáticamente por crontab el 1° de cada mes a las 02:00 AM.
# ==============================================================================

# Si se pasa un año y mes como argumento, usa esos parámetros.
# Por defecto para el entorno de Tesis (datos históricos congelados en 2025), cierra 2025-12.
TARGET_YEAR=${1:-2025}
TARGET_MONTH=${2:-12}

echo "==========================================================================" >> /var/log/tesis_cierre_mensual.log
echo "[CRON] Ejecutando Cierre Mensual automático: $TARGET_YEAR-$TARGET_MONTH a las $(date)" >> /var/log/tesis_cierre_mensual.log

# Disparar endpoint de cierre en el contenedor Docker en puerto 8001
RESPONSE=$(curl -s -X POST "http://localhost:8001/pipeline/run-cierre-mensual" \
     -H "Content-Type: application/json" \
     -d "{\"anio\": $TARGET_YEAR, \"mes\": $TARGET_MONTH}")

echo "[CRON] Respuesta de la API: $RESPONSE" >> /var/log/tesis_cierre_mensual.log
echo "[CRON] Cierre mensual finalizado con éxito a las $(date)" >> /var/log/tesis_cierre_mensual.log
echo "==========================================================================" >> /var/log/tesis_cierre_mensual.log
