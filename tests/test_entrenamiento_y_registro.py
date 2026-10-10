"""Pruebas de entrenamiento, registro de versiones y explicabilidad del modelo Champion.

Estos metodos dependen de MySQL en producción, pero su logica (ventana de
validacion, seleccion de familia de modelo, armado de hiperparametros, calculo
de importancia de atributos) es independiente de la base concreta. Se simula el
motor (``self.engine``) para ejercitar esa logica sin una base real, siguiendo el
mismo criterio que ya usan ``test_features_derivadas.py`` y
``test_target_relativo.py`` para las piezas puras del mismo modulo.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

import tesis_ia.training.db_trainer_predictor as dtp
from tesis_ia.training.db_trainer_predictor import DbTrainerPredictor


@pytest.fixture(autouse=True)
def limpiar_cache_de_modelo():
    """El modulo cachea el modelo entrenado en variables globales de proceso.

    Sin este fixture, un test que entrena un modelo dejaria su cache contaminando
    al siguiente test que dependa de ``_CACHED_MODEL`` siendo ``None``.
    """
    dtp._CACHED_MODEL = None
    dtp._CACHED_RATIO_MODEL = None
    yield
    dtp._CACHED_MODEL = None
    dtp._CACHED_RATIO_MODEL = None


def dataset_sintetico(n_meses=30, productos=("SKU-1", "SKU-2", "SKU-3"), id_empresa="EMP-1"):
    """Un dataset con la forma exacta que devuelve ``load_dataset_from_db``.

    Ventas con tendencia leve y estacionalidad simple, suficientes meses para
    que la ventana movil de 12 meses de validacion tenga datos de ambos lados
    del corte (``train_mask`` y ``test_mask`` no vacios).
    """
    filas = []
    rng = np.random.default_rng(42)
    for prod_idx, producto in enumerate(productos):
        nivel = 50.0 + prod_idx * 20.0
        for mes_idx in range(n_meses):
            anio = 2022 + (mes_idx // 12)
            mes = (mes_idx % 12) + 1
            unidades = max(1.0, nivel + 5.0 * np.sin(mes_idx / 3.0) + rng.normal(0, 3))
            filas.append({
                "id_venta": f"{producto}-{mes_idx}",
                "id_producto": producto,
                "id_empresa": id_empresa,
                "anio": anio,
                "mes": mes,
                "unidades_vendidas": unidades,
                "monto_facturado": unidades * 1000.0,
                "precio_unitario": 1000.0 + rng.normal(0, 10),
                "lag_1": unidades,
                "lag_2": unidades,
                "lag_12": unidades,
                "ma_3m": unidades,
                "market_share_unidades": 0.1,
                "price_gap_pct": 0.0,
                "temp_max_promedio": 30.0,
                "temp_media_promedio": 25.0,
                "precipitacion_mm": 100.0,
                "ipc_indice": 100.0 + mes_idx * 0.5,
                "ipc_var_interanual": 4.5,
                "tipo_cambio_pyg_usd": 7200.0,
                "cantidad_feriados": 0,
                "tiene_festivo_comercial": 0,
            })
    return pd.DataFrame(filas)


@pytest.fixture
def predictor():
    return DbTrainerPredictor(engine=MagicMock(name="engine"))


class TestLoadDatasetFromDb:
    def test_devuelve_el_resultado_de_la_consulta_sql(self, predictor, monkeypatch):
        esperado = dataset_sintetico(n_meses=3, productos=("SKU-1",))
        conn = MagicMock()
        predictor.engine.connect.return_value.__enter__.return_value = conn
        monkeypatch.setattr(dtp.pd, "read_sql", MagicMock(return_value=esperado))

        resultado = predictor.load_dataset_from_db()

        assert resultado is esperado
        predictor.engine.connect.assert_called_once()

    def test_la_consulta_cruza_ventas_con_las_tres_tablas_exogenas(self, predictor, monkeypatch):
        capturada = {}

        def read_sql_falso(query, conn):
            capturada["query"] = query
            return pd.DataFrame()

        predictor.engine.connect.return_value.__enter__.return_value = MagicMock()
        monkeypatch.setattr(dtp.pd, "read_sql", read_sql_falso)

        predictor.load_dataset_from_db()

        for tabla in ("ventas_mensuales", "variables_climaticas", "variables_macroeconomicas", "variables_calendario"):
            assert tabla in capturada["query"]


class TestTrainAndEvaluate:
    def test_entrena_random_forest_por_defecto_y_cachea_ambos_modelos(self, predictor, monkeypatch):
        df = dataset_sintetico()
        monkeypatch.setattr(predictor, "load_dataset_from_db", lambda: df.copy())

        model, metrics, desde, hasta = predictor.train_and_evaluate()

        assert hasattr(model, "estimators_")
        assert set(metrics) == {"mae", "rmse", "mape"}
        assert all(v >= 0 for v in metrics.values())
        assert desde == "2022-01-01"
        assert dtp._CACHED_MODEL is model
        assert dtp._CACHED_RATIO_MODEL is not None

    def test_el_baseline_de_regresion_lineal_no_entrena_especialista_de_ratio(self, predictor, monkeypatch):
        df = dataset_sintetico()
        monkeypatch.setattr(predictor, "load_dataset_from_db", lambda: df.copy())

        model, metrics, _, _ = predictor.train_and_evaluate(model_family="linear_regression")

        assert not hasattr(model, "estimators_")
        assert dtp._CACHED_MODEL is model
        assert dtp._CACHED_RATIO_MODEL is None

    def test_recurre_a_una_ventana_de_seis_meses_si_hay_menos_de_veinticuatro(self, predictor, monkeypatch):
        # Con pocos meses de historia, 12 meses de validacion dejarian el set de
        # entrenamiento vacio; el codigo cae a una ventana de 6 meses.
        df = dataset_sintetico(n_meses=10, productos=("SKU-1",))
        monkeypatch.setattr(predictor, "load_dataset_from_db", lambda: df.copy())

        model, metrics, _, _ = predictor.train_and_evaluate()

        assert model is not None
        assert metrics["mae"] >= 0

    def test_rechaza_un_dataset_vacio(self, predictor, monkeypatch):
        monkeypatch.setattr(predictor, "load_dataset_from_db", lambda: pd.DataFrame())

        with pytest.raises(ValueError, match="No hay datos"):
            predictor.train_and_evaluate()

    def test_el_mape_ignora_los_meses_con_cero_unidades_reales(self, predictor, monkeypatch):
        # Division por cero evitada explicitamente: un mes con unidades_vendidas=0
        # no puede aportar al denominador del MAPE.
        df = dataset_sintetico(n_meses=24, productos=("SKU-1",))
        df.loc[df.index[-1], "unidades_vendidas"] = 0.0
        monkeypatch.setattr(predictor, "load_dataset_from_db", lambda: df.copy())

        _, metrics, _, _ = predictor.train_and_evaluate()

        assert np.isfinite(metrics["mape"])


class TestSaveModelVersion:
    def _conn_falsa(self, empresas_activas, modelos_activos):
        conn = MagicMock()

        def execute(stmt, params=None):
            texto = str(stmt)
            resultado = MagicMock()
            if "SELECT id FROM empresa" in texto:
                resultado.fetchall.return_value = [(e,) for e in empresas_activas]
            elif "SELECT id_empresa, id FROM modelos_versiones" in texto:
                resultado.fetchall.return_value = list(modelos_activos)
            return resultado

        conn.execute.side_effect = execute
        return conn

    def test_registra_el_modelo_para_cada_empresa_activa(self, predictor):
        conn = self._conn_falsa(
            empresas_activas=["EMP-1", "EMP-2"],
            modelos_activos=[("EMP-1", "model-a"), ("EMP-2", "model-b")],
        )
        predictor.engine.begin.return_value.__enter__.return_value = conn

        mapa = predictor.save_model_version(
            version="champion_rf_test", metrics={"mae": 1.0, "rmse": 2.0, "mape": 3.0},
            desde="2024-01-01", hasta="2024-12-31",
        )

        assert mapa == {"EMP-1": "model-a", "EMP-2": "model-b"}
        assert conn.execute.call_count >= 1 + 2 * 2

    def test_registra_solo_la_empresa_indicada_cuando_se_pasa_id_empresa(self, predictor):
        conn = self._conn_falsa(empresas_activas=[], modelos_activos=[("EMP-7", "model-x")])
        predictor.engine.begin.return_value.__enter__.return_value = conn

        mapa = predictor.save_model_version(
            version="v1", metrics={"mae": 1.0, "rmse": 1.0, "mape": 1.0},
            desde="2024-01-01", hasta="2024-12-31", id_empresa="EMP-7",
        )

        # No debe consultar la lista de empresas activas: se le dio una sola.
        consultas = [str(c.args[0]) for c in conn.execute.call_args_list]
        assert not any("SELECT id FROM empresa" in q for q in consultas)
        assert mapa == {"EMP-7": "model-x"}

    def test_usa_hiperparametros_distintos_segun_la_familia_del_modelo(self, predictor, monkeypatch):
        capturados = []
        conn = self._conn_falsa(empresas_activas=["EMP-1"], modelos_activos=[("EMP-1", "m")])
        original_execute = conn.execute.side_effect

        def execute_y_capturar(stmt, params=None):
            if params and "hiperparametros" in params:
                capturados.append(params["hiperparametros"])
            return original_execute(stmt, params)

        conn.execute.side_effect = execute_y_capturar
        predictor.engine.begin.return_value.__enter__.return_value = conn

        predictor.save_model_version(
            version="v1", metrics={"mae": 1.0, "rmse": 1.0, "mape": 1.0},
            desde="2024-01-01", hasta="2024-12-31", model_family="linear_regression",
        )

        assert any("LinearRegression" in c for c in capturados)


class TestGetModelExplainability:
    def test_normaliza_la_importancia_de_atributos_a_cien_por_ciento(self, predictor):
        modelo = MagicMock()
        modelo.feature_importances_ = np.array([4.0, 3.0, 2.0, 1.0] + [0.0] * 8)
        dtp._CACHED_MODEL = modelo

        resultado = predictor.get_model_explainability()

        assert resultado["status"] == "success"
        total = sum(v["importancia_pct"] for v in resultado["importancia_variables"])
        assert total == pytest.approx(100.0, abs=0.1)
        importancias = [v["importancia_pct"] for v in resultado["importancia_variables"]]
        assert importancias == sorted(importancias, reverse=True)

    def test_usa_el_valor_absoluto_de_los_coeficientes_para_un_modelo_lineal(self, predictor):
        modelo = MagicMock(spec=["coef_"])
        modelo.coef_ = np.array([-5.0, 5.0] + [0.0] * 10)
        dtp._CACHED_MODEL = modelo

        resultado = predictor.get_model_explainability()

        top_dos = {v["feature"] for v in resultado["importancia_variables"][:2]}
        assert top_dos == {"lag_1", "lag_2"}

    def test_reparte_importancia_uniforme_si_el_modelo_no_expone_ninguna(self, predictor):
        dtp._CACHED_MODEL = MagicMock(spec=[])

        resultado = predictor.get_model_explainability()

        importancias = [v["importancia_pct"] for v in resultado["importancia_variables"]]
        assert importancias == pytest.approx(importancias[0:1] * len(importancias))

    def test_entrena_si_no_hay_modelo_en_cache(self, predictor, monkeypatch):
        df = dataset_sintetico(n_meses=15, productos=("SKU-1",))
        monkeypatch.setattr(predictor, "load_dataset_from_db", lambda: df.copy())
        assert dtp._CACHED_MODEL is None

        resultado = predictor.get_model_explainability()

        assert resultado["status"] == "success"
        assert dtp._CACHED_MODEL is not None
