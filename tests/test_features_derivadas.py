"""Pruebas de las features del especialista de ratio.

El valor de este archivo esta concentrado en una sola idea: market_share_unidades
y price_gap_pct del mes t se calculan con las ventas de ese mismo mes. Usarlas sin
desplazar seria mostrarle al modelo la respuesta que tiene que predecir. El
desplazamiento de un mes es lo unico que separa a este proyecto de un modelo con
metricas excelentes y ningun valor predictivo, y es exactamente el tipo de detalle
que una refactorizacion bien intencionada elimina sin darse cuenta.
"""

import numpy as np
import pandas as pd
import pytest

from tesis_ia.training.db_trainer_predictor import DbTrainerPredictor


@pytest.fixture
def predictor():
    """Instancia sin conexion: derive_extra_features no consulta la base."""
    return DbTrainerPredictor(engine=object())


def fila_de_ventas(**cambios):
    """Una fila de ventas_mensuales con todas las columnas que la funcion exige."""
    base = {
        "id_producto": "SKU-1",
        "id_empresa": 7,
        "anio": 2025,
        "mes": 1,
        "precio_unitario": 1000.0,
        "price_gap_pct": 0.0,
        "market_share_unidades": 0.0,
        "ma_3m": 30.0,
        "lag_1": 30.0,
        "lag_12": 30.0,
    }
    base.update(cambios)
    return base


def serie_mensual(meses, **columnas_por_mes):
    """Construye un SKU con una fila por mes y columnas variables entre meses."""
    filas = []
    for i, mes in enumerate(meses):
        cambios = {clave: valores[i] for clave, valores in columnas_por_mes.items()}
        filas.append(fila_de_ventas(mes=mes, **cambios))
    return pd.DataFrame(filas)


class TestAusenciaDeFugaDeDatos:

    def test_la_participacion_de_mercado_entra_desplazada_un_mes(self, predictor):
        df = serie_mensual([1, 2, 3], market_share_unidades=[0.10, 0.20, 0.30])

        resultado = predictor.derive_extra_features(df)

        # Cada mes ve la participacion del mes anterior, nunca la propia.
        assert resultado["market_share_lag1"].tolist() == pytest.approx([0.0, 0.10, 0.20])

    def test_la_brecha_de_precio_entra_desplazada_un_mes(self, predictor):
        df = serie_mensual([1, 2, 3], price_gap_pct=[5.0, -3.0, 8.0])

        resultado = predictor.derive_extra_features(df)

        assert resultado["price_gap_lag1"].tolist() == pytest.approx([0.0, 5.0, -3.0])

    def test_el_precio_unitario_entra_desplazado_un_mes(self, predictor):
        df = serie_mensual([1, 2, 3], precio_unitario=[1000.0, 1100.0, 1250.0])

        resultado = predictor.derive_extra_features(df)

        assert resultado["precio_unitario_lag1"].tolist() == pytest.approx(
            [0.0, 1000.0, 1100.0]
        )

    def test_ninguna_variable_contemporanea_coincide_con_su_version_desplazada(
        self, predictor
    ):
        # Comprobacion directa de la fuga: si alguien quitara el shift, estas tres
        # columnas quedarian identicas a las originales y la prueba fallaria.
        df = serie_mensual(
            [1, 2, 3],
            market_share_unidades=[0.10, 0.20, 0.30],
            price_gap_pct=[5.0, -3.0, 8.0],
            precio_unitario=[1000.0, 1100.0, 1250.0],
        )

        resultado = predictor.derive_extra_features(df)

        for contemporanea, desplazada in [
            ("market_share_unidades", "market_share_lag1"),
            ("price_gap_pct", "price_gap_lag1"),
            ("precio_unitario", "precio_unitario_lag1"),
        ]:
            assert not np.allclose(
                resultado[contemporanea].to_numpy(dtype=float),
                resultado[desplazada].to_numpy(dtype=float),
            ), f"{desplazada} no esta desplazada respecto de {contemporanea}"

    def test_el_primer_mes_del_sku_no_hereda_valores_de_otro_producto(self, predictor):
        # El desplazamiento se calcula por grupo. Si se hiciera sobre el dataset
        # entero, el primer mes de un SKU tomaria el ultimo mes del SKU anterior.
        primero = serie_mensual([1, 2], market_share_unidades=[0.10, 0.20])
        segundo = serie_mensual([1, 2], market_share_unidades=[0.70, 0.80])
        segundo["id_producto"] = "SKU-2"

        resultado = predictor.derive_extra_features(pd.concat([primero, segundo]))

        arranque = resultado[resultado["mes"] == 1]["market_share_lag1"]
        assert arranque.tolist() == pytest.approx([0.0, 0.0])

    def test_el_desplazamiento_no_cruza_empresas(self, predictor):
        # El mismo SKU puede venderse en varias empresas: son series distintas y el
        # aislamiento entre inquilinos tiene que sostenerse tambien en las features.
        empresa_a = serie_mensual([1, 2], market_share_unidades=[0.10, 0.20])
        empresa_b = serie_mensual([1, 2], market_share_unidades=[0.70, 0.80])
        empresa_b["id_empresa"] = 9

        resultado = predictor.derive_extra_features(pd.concat([empresa_a, empresa_b]))

        de_b = resultado[resultado["id_empresa"] == 9].sort_values("mes")
        assert de_b["market_share_lag1"].tolist() == pytest.approx([0.0, 0.70])

    def test_desplaza_en_orden_cronologico_aunque_las_filas_lleguen_desordenadas(
        self, predictor
    ):
        df = serie_mensual([3, 1, 2], market_share_unidades=[0.30, 0.10, 0.20])

        resultado = predictor.derive_extra_features(df)

        assert resultado["mes"].tolist() == [1, 2, 3]
        assert resultado["market_share_lag1"].tolist() == pytest.approx([0.0, 0.10, 0.20])


class TestEstacionalidadCiclica:
    """Hasta esta version el mes solo entraba al modelo a traves de lag_12."""

    def test_codifica_el_mes_sobre_la_circunferencia_unitaria(self, predictor):
        df = serie_mensual([3, 6, 9, 12])

        resultado = predictor.derive_extra_features(df)

        assert resultado["mes_sin"].tolist() == pytest.approx([1.0, 0.0, -1.0, 0.0], abs=1e-12)
        assert resultado["mes_cos"].tolist() == pytest.approx([0.0, -1.0, 0.0, 1.0], abs=1e-12)

    def test_diciembre_y_enero_quedan_contiguos(self, predictor):
        df = serie_mensual([1, 6, 12])

        resultado = predictor.derive_extra_features(df).set_index("mes")

        def distancia(mes_a, mes_b):
            return np.hypot(
                resultado.loc[mes_a, "mes_sin"] - resultado.loc[mes_b, "mes_sin"],
                resultado.loc[mes_a, "mes_cos"] - resultado.loc[mes_b, "mes_cos"],
            )

        assert distancia(12, 1) < distancia(1, 6)

    def test_todos_los_meses_caen_sobre_la_circunferencia(self, predictor):
        df = serie_mensual(list(range(1, 13)))

        resultado = predictor.derive_extra_features(df)

        radio = np.hypot(resultado["mes_sin"], resultado["mes_cos"])
        assert radio.tolist() == pytest.approx([1.0] * 12)


class TestEscalaDeReferencia:

    def test_log_base_es_el_logaritmo_de_la_base_de_escala(self, predictor):
        # El especialista aprende log1p(y) - log_base, de modo que esta columna es
        # la que reconstruye el nivel absoluto en inferencia.
        df = serie_mensual([1, 2], ma_3m=[30.0, 0.0], lag_1=[30.0, 12.0])

        resultado = predictor.derive_extra_features(df)

        assert resultado["log_base"].tolist() == pytest.approx(
            np.log1p([30.0, 12.0]).tolist()
        )

    def test_aplica_la_misma_cascada_que_calcular_base_escala(self, predictor):
        # Entrenamiento e inferencia comparten esta funcion: si divergieran, el
        # target relativo se reconstruiria contra una escala distinta.
        df = serie_mensual(
            [1, 2, 3], ma_3m=[30.0, 0.0, 0.0], lag_1=[10.0, 10.0, 0.0], lag_12=[20.0, 20.0, 20.0]
        )

        resultado = predictor.derive_extra_features(df)

        esperada = DbTrainerPredictor.calcular_base_escala(
            df.sort_values("mes")["ma_3m"],
            df.sort_values("mes")["lag_1"],
            df.sort_values("mes")["lag_12"],
        )
        assert resultado["base_escala"].tolist() == pytest.approx(list(esperada))

    def test_un_sku_sin_historia_queda_con_escala_y_logaritmo_nulos(self, predictor):
        df = serie_mensual([1], ma_3m=[0.0], lag_1=[0.0], lag_12=[0.0])

        resultado = predictor.derive_extra_features(df)

        assert resultado["base_escala"].tolist() == pytest.approx([0.0])
        assert resultado["log_base"].tolist() == pytest.approx([0.0])


class TestContratoDelDataset:

    def test_produce_todas_las_columnas_que_el_especialista_consume(self, predictor):
        # EXTRA_FEATURE_COLS es lo que se le pasa al modelo de ratio: si la funcion
        # dejara de derivar alguna, el entrenamiento falla con un KeyError opaco.
        resultado = predictor.derive_extra_features(serie_mensual([1, 2, 3]))

        faltantes = set(DbTrainerPredictor.EXTRA_FEATURE_COLS) - set(resultado.columns)
        assert faltantes == set()

    def test_conserva_las_columnas_originales(self, predictor):
        df = serie_mensual([1, 2])

        resultado = predictor.derive_extra_features(df)

        assert set(df.columns).issubset(set(resultado.columns))

    def test_no_agrega_ni_descarta_filas(self, predictor):
        df = serie_mensual([1, 2, 3])

        resultado = predictor.derive_extra_features(df)

        assert len(resultado) == len(df)

    def test_ninguna_feature_derivada_queda_indefinida(self, predictor):
        resultado = predictor.derive_extra_features(serie_mensual([1, 2, 3]))

        derivadas = resultado[DbTrainerPredictor.EXTRA_FEATURE_COLS]
        assert not derivadas.isna().to_numpy().any()
