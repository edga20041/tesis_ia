"""API REST FastAPI para el microservicio de IA de Inteligencia Predictiva."""

from __future__ import annotations

import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tesis_ia.data.exogenous_pipeline import ExogenousPipeline
from tesis_ia.data.feature_calculator import FeatureCalculator
from tesis_ia.models.cold_start import get_cold_start_engine
from tesis_ia.training.db_trainer_predictor import DbTrainerPredictor
from tesis_ia.utils.db_connector import test_connection

app = FastAPI(
    title="Plataforma de Inteligencia Predictiva B2B - API de IA",
    description="Microservicio de entrenamiento, ingesta exógena, predicciones, simulador de escenarios y Cold Start para productos nuevos a 12 meses.",
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


class ColdStartRequest(BaseModel):
    id_departamento: int = Field(..., description="ID del departamento del producto (ej: 1 para Alimentos, 2 para Bebidas)")
    id_categoria: int = Field(..., description="ID de la categoría del producto (ej: 5 para Aceites)")
    id_marca: str = Field(..., description="ID o UUID de la marca del producto")
    descripcion: str = Field("Nuevo Producto", description="Descripción comercial del producto nuevo")
    contenido_neto: float = Field(..., gt=0, description="Contenido neto numérico (ej: 900, 1.5, 500)")
    unidad_medida: str = Field(..., description="Unidad de medida: 'ml', 'lt', 'gr', 'kg', 'un'")
    valor_unitario: float = Field(..., gt=0, description="Precio de venta unitario proyectado en Guaraníes (PYG)")
    horizonte_meses: Optional[int] = Field(12, ge=1, le=24, description="Cantidad de meses a proyectar (def: 12)")
    anio_inicio: Optional[int] = Field(None, description="Año inicial de la proyección (def: año actual)")
    mes_inicio: Optional[int] = Field(None, description="Mes inicial de la proyección (1-12, def: mes actual)")
    id_empresa: Optional[str] = Field(None, description="ID de la empresa (opcional)")


class ColdStartResponse(BaseModel):
    status: str
    producto_nuevo: Dict[str, Any]
    productos_similares_gemelos: List[Dict[str, Any]]
    proyeccion_12_meses: List[Dict[str, Any]]


@app.on_event("startup")
def startup_event():
    try:
        print("[STARTUP] Precargando modelo Champion (Random Forest) en memoria RAM...")
        trainer = DbTrainerPredictor()
        trainer.train_and_evaluate(model_family="random_forest")
        print("[STARTUP] Modelo Champion precargado exitosamente en RAM. Simulaciones de escenarios listas en < 50ms.")
    except Exception as e:
        print(f"[WARN] No se pudo precargar el modelo Champion en startup: {e}")

    try:
        print("[STARTUP] Precargando motor de Cold Start (K-Means + KNN) en memoria RAM...")
        cold_engine = get_cold_start_engine()
        cold_engine.fit()
        print("[STARTUP] Motor de Cold Start inicializado exitosamente en RAM.")
    except Exception as e:
        print(f"[WARN] No se pudo inicializar Cold Start en startup: {e}")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    db_ok = test_connection()
    return {
        "status": "healthy" if db_ok else "unhealthy",
        "database_connected": db_ok,
        "timestamp": datetime.datetime.now().isoformat(),
    }


def _ejecutar_pipeline_cierre(anio: int, mes: int, id_empresa: Optional[str]) -> Dict[str, Any]:
    print(f"\n=== Iniciando Pipeline de Cierre Mensual para {anio}-{mes:02d} ===")

    exo_pipe = ExogenousPipeline()
    exo_ok, exo_details = exo_pipe.sync_month(anio, mes)
    if not exo_ok:
        print("[WARN] Falló la ingesta exógena, continuando con valores previos.")

    feat_calc = FeatureCalculator()
    feat_calc.consolidate_weekly_sales(anio, mes)
    feat_calc.compute_features_month(anio, mes)

    trainer = DbTrainerPredictor()
    model, metrics, desde, hasta = trainer.train_and_evaluate(model_family="random_forest")

    version_name = f"champion_rf_{anio}_{mes:02d}"
    company_model_map = trainer.save_model_version(version_name, metrics, desde, hasta, id_empresa, model_family="random_forest")

    num_preds = trainer.generate_and_save_predictions(model, company_model_map)
    first_model_id = next(iter(company_model_map.values()), "model_id")

    return {
        "status": "success",
        "message": f"Pipeline de cierre mensual para {anio}-{mes:02d} ejecutado correctamente con modelo Champion (Random Forest).",
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


@app.post("/cold-start/predict", response_model=ColdStartResponse)
def predict_cold_start(req: ColdStartRequest) -> ColdStartResponse:
    try:
        engine = get_cold_start_engine()
        res = engine.predict_new_product(
            id_departamento=req.id_departamento,
            id_categoria=req.id_categoria,
            id_marca=req.id_marca,
            descripcion=req.descripcion,
            contenido_neto=req.contenido_neto,
            unidad_medida=req.unidad_medida,
            valor_unitario=req.valor_unitario,
            horizonte_meses=req.horizonte_meses or 12,
            anio_inicio=req.anio_inicio,
            mes_inicio=req.mes_inicio,
            id_empresa=req.id_empresa,
        )
        return ColdStartResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción de Cold Start: {str(e)}")


@app.get("/model/explainability")
def get_model_explainability() -> Dict[str, Any]:
    try:
        trainer = DbTrainerPredictor()
        return trainer.get_model_explainability()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener explicabilidad del modelo: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tesis_ia.api:app", host="0.0.0.0", port=8000, reload=True)
