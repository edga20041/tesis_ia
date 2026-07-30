"""API REST FastAPI para el microservicio de IA de Inteligencia Predictiva."""

from __future__ import annotations

import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tesis_ia.data.exogenous_pipeline import ExogenousPipeline
from tesis_ia.data.feature_calculator import FeatureCalculator
from tesis_ia.training.db_trainer_predictor import DbTrainerPredictor
from tesis_ia.utils.db_connector import test_connection

app = FastAPI(
    title="Plataforma de Inteligencia Predictiva B2B - API de IA",
    description="Microservicio de entrenamiento, ingesta exógena, predicciones y simulador de escenarios a 12 meses.",
    version="1.0.0",
)


class CierreMensualRequest(BaseModel):
    anio: int = Field(..., ge=2000, le=2100, description="Año del cierre mensual")
    mes: int = Field(..., ge=1, le=12, description="Mes del cierre mensual (1-12)")
    id_empresa: Optional[str] = Field(None, description="ID de la empresa (opcional)")


class PipelineResponse(BaseModel):
    status: str
    message: str
    anio: int
    mes: int
    exogenas_sincronizadas: Optional[Dict[str, Any]] = None
    id_modelo: Optional[str] = None
    predicciones_generadas: int = 0


class SimulationRequest(BaseModel):
    id_producto: str = Field(..., description="ID o SKU del producto a simular")
    precio_pct: float = Field(0.0, description="Variación porcentual del precio (-30 a +30)")
    usd_pct: float = Field(0.0, description="Variación porcentual del Dólar USD (-20 a +20)")
    ipc_pct: float = Field(0.0, description="Variación porcentual del IPC Inflación (-10 a +20)")


class SimulationResponse(BaseModel):
    status: str
    id_producto: str
    variaciones_aplicadas: Dict[str, float]
    simulacion: List[Dict[str, Any]]


@app.get("/health")
def health_check() -> Dict[str, Any]:
    db_ok = test_connection()
    return {
        "status": "healthy" if db_ok else "unhealthy",
        "database_connected": db_ok,
        "timestamp": datetime.datetime.now().isoformat(),
    }


def _ejecutar_pipeline_cierre(anio: int, mes: int, id_empresa: Optional[str]) -> Dict[str, Any]:
    """Ejecuta sincrónicamente el ciclo completo de cierre mensual."""
    print(f"\n=== Iniciando Pipeline de Cierre Mensual para {anio}-{mes:02d} ===")

    exo_pipe = ExogenousPipeline()
    exo_ok, exo_details = exo_pipe.sync_month(anio, mes)
    if not exo_ok:
        print("[WARN] Falló la ingesta exógena, continuando con valores previos.")

    feat_calc = FeatureCalculator()
    feat_calc.consolidate_weekly_sales(anio, mes)
    feat_calc.compute_features_month(anio, mes)

    trainer = DbTrainerPredictor()
    model, metrics, desde, hasta = trainer.train_and_evaluate()

    version_name = f"baseline_linear_{anio}_{mes:02d}"
    company_model_map = trainer.save_model_version(version_name, metrics, desde, hasta, id_empresa)

    num_preds = trainer.generate_and_save_predictions(model, company_model_map)
    first_model_id = next(iter(company_model_map.values()), "model_id")

    return {
        "status": "success",
        "message": f"Pipeline de cierre mensual para {anio}-{mes:02d} ejecutado correctamente.",
        "anio": anio,
        "mes": mes,
        "exogenas_sincronizadas": exo_details,
        "id_modelo": first_model_id,
        "predicciones_generadas": num_preds,
    }


@app.post("/pipeline/run-cierre-mensual", response_model=PipelineResponse)
def run_cierre_mensual(req: CierreMensualRequest) -> PipelineResponse:
    try:
        res = _ejecutar_pipeline_cierre(req.anio, req.mes, req.id_empresa)
        return PipelineResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el cierre mensual: {str(e)}")


@app.post("/simulation/predict", response_model=SimulationResponse)
def run_simulation(req: SimulationRequest) -> SimulationResponse:
    try:
        trainer = DbTrainerPredictor()
        simulados = trainer.simulate_product_scenarios(
            id_producto=req.id_producto,
            precio_pct=req.precio_pct,
            usd_pct=req.usd_pct,
            ipc_pct=req.ipc_pct,
        )
        return SimulationResponse(
            status="success",
            id_producto=req.id_producto,
            variaciones_aplicadas={
                "precio_pct": req.precio_pct,
                "usd_pct": req.usd_pct,
                "ipc_pct": req.ipc_pct,
            },
            simulacion=simulados,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ejecutar la simulación de producto: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tesis_ia.api:app", host="0.0.0.0", port=8000, reload=True)
