"""Pruebas de la capa REST del microservicio de IA.

api.py es la unica puerta de entrada al sistema de pronostico: el backend
transaccional la consume por HTTP y de ella dependen el cierre mensual, el
simulador what-if, el arranque en frio y la explicabilidad. Su logica propia es
delgada -validar el contrato, delegar y traducir fallas a HTTP- pero es
justamente lo que el consumidor ve, asi que un cambio silencioso en el contrato
rompe la integracion sin que ninguna otra suite lo advierta.

Todo lo que toca la base o entrena modelos (DbTrainerPredictor, el motor de
arranque en frio, el pipeline exogeno y el calculador de features) se sustituye
por dobles en el espacio de nombres de tesis_ia.api. Lo que se ejercita tal cual
es FastAPI: el ruteo, la validacion de pydantic y la serializacion de respuestas.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tesis_ia import api


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def trainer(monkeypatch):
    """DbTrainerPredictor sustituido; devuelve la instancia que la API usara."""
    instancia = MagicMock(name="DbTrainerPredictor()")
    monkeypatch.setattr(api, "DbTrainerPredictor", MagicMock(return_value=instancia))
    return instancia


@pytest.fixture
def cold_engine(monkeypatch):
    engine = MagicMock(name="ColdStartEngine")
    monkeypatch.setattr(api, "get_cold_start_engine", MagicMock(return_value=engine))
    return engine


@pytest.fixture
def pipeline(monkeypatch, trainer):
    """Cierre mensual completo con todas sus dependencias sustituidas."""
    exo = MagicMock(name="ExogenousPipeline()")
    exo.sync_month.return_value = (True, {"clima": "ok", "macro": "ok"})
    feat = MagicMock(name="FeatureCalculator()")
    monkeypatch.setattr(api, "ExogenousPipeline", MagicMock(return_value=exo))
    monkeypatch.setattr(api, "FeatureCalculator", MagicMock(return_value=feat))

    trainer.train_and_evaluate.return_value = ("modelo", {"mape": 15.66}, "2021-01", "2025-12")
    trainer.save_model_version.return_value = {"empresa-1": "modelo-id-1", "empresa-2": "modelo-id-2"}
    trainer.generate_and_save_predictions.return_value = 285660
    return {"exo": exo, "feat": feat, "trainer": trainer}


CIERRE = {"anio": 2025, "mes": 12}

COLD_START = {
    "id_departamento": 1,
    "id_categoria": 5,
    "id_marca": "marca-1",
    "descripcion": "Aceite de girasol",
    "contenido_neto": 900,
    "unidad_medida": "ml",
    "valor_unitario": 15000,
}

COLD_START_RESPUESTA = {
    "status": "success",
    "producto_nuevo": {"descripcion": "Aceite de girasol"},
    "productos_similares_gemelos": [{"id_producto": "p1", "afinidad": 0.91}],
    "proyeccion_12_meses": [{"periodo": "2026-01", "unidades": 120.0}],
}


class TestHealth:
    def test_reporta_saludable_cuando_la_base_responde(self, client, monkeypatch):
        monkeypatch.setattr(api, "test_connection", lambda: True)

        r = client.get("/health")

        assert r.status_code == 200
        cuerpo = r.json()
        assert cuerpo["status"] == "healthy"
        assert cuerpo["database_connected"] is True

    def test_reporta_no_saludable_sin_base_pero_responde_200(self, client, monkeypatch):
        # El healthcheck de Docker consulta este endpoint: un servicio que
        # responde con la base caida debe poder informarlo, no dejar de responder.
        monkeypatch.setattr(api, "test_connection", lambda: False)

        r = client.get("/health")

        assert r.status_code == 200
        assert r.json()["status"] == "unhealthy"
        assert r.json()["database_connected"] is False

    def test_incluye_una_marca_de_tiempo_iso(self, client, monkeypatch):
        import datetime

        monkeypatch.setattr(api, "test_connection", lambda: True)

        marca = client.get("/health").json()["timestamp"]

        datetime.datetime.fromisoformat(marca)


class TestCierreMensual:
    def test_ejecuta_las_etapas_del_pipeline_con_el_periodo_pedido(self, client, pipeline):
        r = client.post("/pipeline/run-cierre-mensual", json=CIERRE)

        assert r.status_code == 200
        pipeline["exo"].sync_month.assert_called_once_with(2025, 12)
        pipeline["feat"].consolidate_weekly_sales.assert_called_once_with(2025, 12)
        pipeline["feat"].compute_features_month.assert_called_once_with(2025, 12)
        pipeline["trainer"].train_and_evaluate.assert_called_once_with(model_family="random_forest")

    def test_respeta_el_orden_exogenas_features_entrenamiento_predicciones(self, client, pipeline):
        orden = MagicMock()
        orden.attach_mock(pipeline["exo"].sync_month, "exogenas")
        orden.attach_mock(pipeline["feat"].consolidate_weekly_sales, "consolidar")
        orden.attach_mock(pipeline["feat"].compute_features_month, "features")
        orden.attach_mock(pipeline["trainer"].train_and_evaluate, "entrenar")
        orden.attach_mock(pipeline["trainer"].generate_and_save_predictions, "predecir")

        client.post("/pipeline/run-cierre-mensual", json=CIERRE)

        etapas = [llamada[0] for llamada in orden.mock_calls]
        assert etapas == ["exogenas", "consolidar", "features", "entrenar", "predecir"]

    def test_versiona_el_modelo_con_el_periodo_en_el_nombre(self, client, pipeline):
        client.post("/pipeline/run-cierre-mensual", json={"anio": 2025, "mes": 3})

        args, kwargs = pipeline["trainer"].save_model_version.call_args
        assert args[0] == "champion_rf_2025_03"
        assert args[1] == {"mape": 15.66}
        assert args[2:4] == ("2021-01", "2025-12")
        assert kwargs["model_family"] == "random_forest"

    def test_propaga_la_empresa_opcional_al_versionado(self, client, pipeline):
        client.post("/pipeline/run-cierre-mensual", json={**CIERRE, "id_empresa": "empresa-9"})

        args, _ = pipeline["trainer"].save_model_version.call_args
        assert args[4] == "empresa-9"

    def test_sin_empresa_versiona_para_todas(self, client, pipeline):
        client.post("/pipeline/run-cierre-mensual", json=CIERRE)

        args, _ = pipeline["trainer"].save_model_version.call_args
        assert args[4] is None

    def test_genera_las_predicciones_sobre_el_modelo_entrenado(self, client, pipeline):
        client.post("/pipeline/run-cierre-mensual", json=CIERRE)

        pipeline["trainer"].generate_and_save_predictions.assert_called_once_with(
            "modelo", {"empresa-1": "modelo-id-1", "empresa-2": "modelo-id-2"}
        )

    def test_respuesta_informa_el_resultado_del_cierre(self, client, pipeline):
        r = client.post("/pipeline/run-cierre-mensual", json=CIERRE)

        cuerpo = r.json()
        assert cuerpo["status"] == "success"
        assert cuerpo["anio"] == 2025
        assert cuerpo["mes"] == 12
        assert cuerpo["predicciones_generadas"] == 285660
        assert cuerpo["id_modelo"] == "modelo-id-1"
        assert cuerpo["exogenas_sincronizadas"] == {"clima": "ok", "macro": "ok"}

    def test_una_falla_de_ingesta_exogena_no_detiene_el_cierre(self, client, pipeline):
        # El pipeline continua con los valores previos: perder un mes de clima
        # no justifica dejar sin predicciones a todas las empresas.
        pipeline["exo"].sync_month.return_value = (False, {"error": "API caida"})

        r = client.post("/pipeline/run-cierre-mensual", json=CIERRE)

        assert r.status_code == 200
        pipeline["trainer"].train_and_evaluate.assert_called_once()
        assert r.json()["exogenas_sincronizadas"] == {"error": "API caida"}

    def test_una_falla_en_el_entrenamiento_responde_500_con_detalle(self, client, pipeline):
        pipeline["trainer"].train_and_evaluate.side_effect = RuntimeError("sin memoria")

        r = client.post("/pipeline/run-cierre-mensual", json=CIERRE)

        assert r.status_code == 500
        assert "cierre mensual" in r.json()["detail"]
        assert "sin memoria" in r.json()["detail"]

    def test_una_falla_temprana_no_llega_a_entrenar(self, client, pipeline):
        pipeline["feat"].compute_features_month.side_effect = RuntimeError("tabla bloqueada")

        r = client.post("/pipeline/run-cierre-mensual", json=CIERRE)

        assert r.status_code == 500
        pipeline["trainer"].train_and_evaluate.assert_not_called()

    @pytest.mark.parametrize("cuerpo", [
        {"anio": 2025, "mes": 0},
        {"anio": 2025, "mes": 13},
        {"anio": 1999, "mes": 6},
        {"anio": 2101, "mes": 6},
        {"anio": 2025},
        {"mes": 6},
        {},
        {"anio": "dos mil", "mes": 6},
    ])
    def test_rechaza_periodos_invalidos_con_422_sin_ejecutar_nada(self, client, pipeline, cuerpo):
        r = client.post("/pipeline/run-cierre-mensual", json=cuerpo)

        assert r.status_code == 422
        pipeline["exo"].sync_month.assert_not_called()
        pipeline["trainer"].train_and_evaluate.assert_not_called()

    @pytest.mark.parametrize("anio,mes", [(2000, 1), (2100, 12)])
    def test_acepta_los_limites_del_rango_valido(self, client, pipeline, anio, mes):
        r = client.post("/pipeline/run-cierre-mensual", json={"anio": anio, "mes": mes})

        assert r.status_code == 200


class TestSimulacion:
    def test_delega_al_entrenador_con_las_variaciones_pedidas(self, client, trainer):
        trainer.simulate_product_scenarios.return_value = []

        client.post("/simulation/predict", json={
            "id_producto": "SKU-1", "precio_pct": -10, "usd_pct": 5, "ipc_pct": 2.5,
        })

        trainer.simulate_product_scenarios.assert_called_once_with(
            id_producto="SKU-1", precio_pct=-10.0, usd_pct=5.0, ipc_pct=2.5,
        )

    def test_las_variaciones_omitidas_valen_cero(self, client, trainer):
        trainer.simulate_product_scenarios.return_value = []

        r = client.post("/simulation/predict", json={"id_producto": "SKU-1"})

        assert r.status_code == 200
        assert r.json()["variaciones_aplicadas"] == {"precio_pct": 0.0, "usd_pct": 0.0, "ipc_pct": 0.0}

    def test_devuelve_la_simulacion_y_repite_lo_aplicado(self, client, trainer):
        meses = [{"periodo": "2026-01", "base": 100.0, "simulado": 92.0}]
        trainer.simulate_product_scenarios.return_value = meses

        r = client.post("/simulation/predict", json={"id_producto": "SKU-1", "precio_pct": 10})

        assert r.status_code == 200
        cuerpo = r.json()
        assert cuerpo["status"] == "success"
        assert cuerpo["id_producto"] == "SKU-1"
        assert cuerpo["variaciones_aplicadas"]["precio_pct"] == 10.0
        assert cuerpo["simulacion"] == meses

    def test_requiere_el_producto(self, client, trainer):
        r = client.post("/simulation/predict", json={"precio_pct": 10})

        assert r.status_code == 422
        trainer.simulate_product_scenarios.assert_not_called()

    def test_una_falla_del_simulador_responde_500_con_detalle(self, client, trainer):
        trainer.simulate_product_scenarios.side_effect = ValueError("producto inexistente")

        r = client.post("/simulation/predict", json={"id_producto": "NO-EXISTE"})

        assert r.status_code == 500
        assert "producto inexistente" in r.json()["detail"]


class TestColdStart:
    def test_pasa_los_atributos_del_producto_al_motor(self, client, cold_engine):
        cold_engine.predict_new_product.return_value = COLD_START_RESPUESTA

        client.post("/cold-start/predict", json={
            **COLD_START, "horizonte_meses": 6, "anio_inicio": 2026, "mes_inicio": 2, "id_empresa": "emp-1",
        })

        cold_engine.predict_new_product.assert_called_once_with(
            id_departamento=1, id_categoria=5, id_marca="marca-1",
            descripcion="Aceite de girasol", contenido_neto=900.0, unidad_medida="ml",
            valor_unitario=15000.0, horizonte_meses=6, anio_inicio=2026, mes_inicio=2,
            id_empresa="emp-1",
        )

    def test_el_horizonte_por_defecto_es_de_doce_meses(self, client, cold_engine):
        cold_engine.predict_new_product.return_value = COLD_START_RESPUESTA

        client.post("/cold-start/predict", json=COLD_START)

        _, kwargs = cold_engine.predict_new_product.call_args
        assert kwargs["horizonte_meses"] == 12
        assert kwargs["anio_inicio"] is None
        assert kwargs["mes_inicio"] is None
        assert kwargs["id_empresa"] is None

    def test_un_horizonte_nulo_tambien_se_resuelve_a_doce_meses(self, client, cold_engine):
        cold_engine.predict_new_product.return_value = COLD_START_RESPUESTA

        client.post("/cold-start/predict", json={**COLD_START, "horizonte_meses": None})

        _, kwargs = cold_engine.predict_new_product.call_args
        assert kwargs["horizonte_meses"] == 12

    def test_la_descripcion_omitida_toma_el_valor_por_defecto(self, client, cold_engine):
        cold_engine.predict_new_product.return_value = COLD_START_RESPUESTA
        cuerpo = {k: v for k, v in COLD_START.items() if k != "descripcion"}

        client.post("/cold-start/predict", json=cuerpo)

        _, kwargs = cold_engine.predict_new_product.call_args
        assert kwargs["descripcion"] == "Nuevo Producto"

    def test_devuelve_la_proyeccion_gemelos_y_producto(self, client, cold_engine):
        cold_engine.predict_new_product.return_value = COLD_START_RESPUESTA

        r = client.post("/cold-start/predict", json=COLD_START)

        assert r.status_code == 200
        assert r.json() == COLD_START_RESPUESTA

    def test_ignora_claves_extra_del_motor_sin_romper_el_contrato(self, client, cold_engine):
        # El motor agrega campos de diagnostico (p. ej. cluster_id en los
        # gemelos); el contrato publicado no debe depender de ellos.
        respuesta = {**COLD_START_RESPUESTA, "campo_interno": "x"}
        cold_engine.predict_new_product.return_value = respuesta

        r = client.post("/cold-start/predict", json=COLD_START)

        assert r.status_code == 200
        assert "campo_interno" not in r.json()

    @pytest.mark.parametrize("campo,valor", [
        ("contenido_neto", 0),
        ("contenido_neto", -5),
        ("valor_unitario", 0),
        ("valor_unitario", -1),
        ("horizonte_meses", 0),
        ("horizonte_meses", 25),
    ])
    def test_rechaza_valores_fuera_de_rango_con_422(self, client, cold_engine, campo, valor):
        r = client.post("/cold-start/predict", json={**COLD_START, campo: valor})

        assert r.status_code == 422
        cold_engine.predict_new_product.assert_not_called()

    @pytest.mark.parametrize("campo", [
        "id_departamento", "id_categoria", "id_marca", "contenido_neto", "unidad_medida", "valor_unitario",
    ])
    def test_exige_los_atributos_obligatorios(self, client, cold_engine, campo):
        cuerpo = {k: v for k, v in COLD_START.items() if k != campo}

        r = client.post("/cold-start/predict", json=cuerpo)

        assert r.status_code == 422
        cold_engine.predict_new_product.assert_not_called()

    @pytest.mark.parametrize("horizonte", [1, 24])
    def test_acepta_los_limites_del_horizonte(self, client, cold_engine, horizonte):
        cold_engine.predict_new_product.return_value = COLD_START_RESPUESTA

        r = client.post("/cold-start/predict", json={**COLD_START, "horizonte_meses": horizonte})

        assert r.status_code == 200

    def test_una_falla_del_motor_responde_500_con_detalle(self, client, cold_engine):
        cold_engine.predict_new_product.side_effect = RuntimeError("catalogo vacio")

        r = client.post("/cold-start/predict", json=COLD_START)

        assert r.status_code == 500
        assert "catalogo vacio" in r.json()["detail"]

    def test_una_respuesta_incompleta_del_motor_se_informa_como_500(self, client, cold_engine):
        # Si el motor omite una clave del contrato, el consumidor recibe un
        # error explicito y no un cuerpo a medias.
        cold_engine.predict_new_product.return_value = {"status": "success"}

        r = client.post("/cold-start/predict", json=COLD_START)

        assert r.status_code == 500


class TestExplicabilidad:
    def test_devuelve_tal_cual_lo_que_calcula_el_entrenador(self, client, trainer):
        xai = {"modelo": "champion_rf_2025_12", "importancias": [{"variable": "lag_1", "peso": 0.8184}]}
        trainer.get_model_explainability.return_value = xai

        r = client.get("/model/explainability")

        assert r.status_code == 200
        assert r.json() == xai

    def test_una_falla_responde_500_con_detalle(self, client, trainer):
        trainer.get_model_explainability.side_effect = RuntimeError("sin modelo entrenado")

        r = client.get("/model/explainability")

        assert r.status_code == 500
        assert "sin modelo entrenado" in r.json()["detail"]


class TestArranque:
    def test_precarga_el_champion_y_el_motor_de_arranque_en_frio(self, trainer, cold_engine):
        api.startup_event()

        trainer.train_and_evaluate.assert_called_once_with(model_family="random_forest")
        cold_engine.fit.assert_called_once_with()

    def test_si_falla_el_modelo_igual_se_inicializa_el_arranque_en_frio(self, trainer, cold_engine):
        trainer.train_and_evaluate.side_effect = RuntimeError("base caida")

        api.startup_event()

        cold_engine.fit.assert_called_once_with()

    def test_si_falla_el_arranque_en_frio_el_servicio_arranca_igual(self, trainer, cold_engine):
        cold_engine.fit.side_effect = RuntimeError("catalogo vacio")

        api.startup_event()

        trainer.train_and_evaluate.assert_called_once()

    def test_una_base_caida_en_el_arranque_no_impide_levantar_el_servicio(self, monkeypatch):
        # Con la base inaccesible la precarga falla, pero el contenedor debe
        # quedar arriba para que /health pueda informar el estado.
        monkeypatch.setattr(api, "DbTrainerPredictor", MagicMock(side_effect=RuntimeError("sin conexion")))
        monkeypatch.setattr(api, "get_cold_start_engine", MagicMock(side_effect=RuntimeError("sin conexion")))

        api.startup_event()


class TestContratoOpenApi:
    def test_publica_los_cinco_endpoints_con_su_metodo(self, client):
        rutas = client.get("/openapi.json").json()["paths"]

        assert set(rutas["/health"]) == {"get"}
        assert set(rutas["/pipeline/run-cierre-mensual"]) == {"post"}
        assert set(rutas["/simulation/predict"]) == {"post"}
        assert set(rutas["/cold-start/predict"]) == {"post"}
        assert set(rutas["/model/explainability"]) == {"get"}
        assert len(rutas) == 5

    def test_los_metodos_no_declarados_responden_405(self, client):
        assert client.get("/simulation/predict").status_code == 405
        assert client.post("/health").status_code == 405

    def test_una_ruta_inexistente_responde_404(self, client):
        assert client.get("/no-existe").status_code == 404
