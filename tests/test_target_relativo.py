"""Pruebas del especialista de target relativo y del hibrido con techo.

Es el aporte central de la tesis: el modelo logaritmico no puede extrapolar por
encima del maximo visto en entrenamiento, el especialista de ratio si puede pero
se descontrola cuando la base historica quedo obsoleta, y el techo relativo acota
ese modo de falla. Las tres piezas que se verifican aca son puras: no tocan la
base de datos ni entrenan nada.
"""

import numpy as np
import pandas as pd
import pytest

from tesis_ia.training.db_trainer_predictor import DbTrainerPredictor


class TestCalcularBaseEscala:
    """La escala de referencia del SKU.

    Su contrato es una cascada: media movil, y si no hay, el mes anterior, y si
    tampoco, el mismo mes del anio pasado. Tiene que calcularse identico en
    entrenamiento y en inferencia, porque el target relativo se reconstruye
    multiplicando por esta base: si difieren, el modelo predice sobre una escala
    distinta a la que aprendio.
    """

    def test_usa_la_media_movil_cuando_la_serie_es_larga(self):
        base = DbTrainerPredictor.calcular_base_escala([30.0], [10.0], [20.0])
        assert base == pytest.approx([30.0])

    def test_cae_al_mes_anterior_cuando_no_hay_media_movil(self):
        base = DbTrainerPredictor.calcular_base_escala([0.0], [10.0], [20.0])
        assert base == pytest.approx([10.0])

    def test_cae_al_mismo_mes_del_anio_pasado_cuando_tampoco_hay_mes_anterior(self):
        # Reingreso de un producto discontinuado: sin ventas recientes, pero con
        # historia del anio anterior.
        base = DbTrainerPredictor.calcular_base_escala([0.0], [0.0], [20.0])
        assert base == pytest.approx([20.0])

    def test_es_cero_cuando_el_sku_no_tiene_ninguna_historia(self):
        base = DbTrainerPredictor.calcular_base_escala([0.0], [0.0], [0.0])
        assert base == pytest.approx([0.0])

    def test_resuelve_la_cascada_elemento_por_elemento(self):
        base = DbTrainerPredictor.calcular_base_escala(
            ma_3m=[30.0, 0.0, 0.0, 0.0],
            lag_1=[10.0, 10.0, 0.0, 0.0],
            lag_12=[20.0, 20.0, 20.0, 0.0],
        )
        assert base == pytest.approx([30.0, 10.0, 20.0, 0.0])

    def test_nunca_devuelve_una_escala_negativa(self):
        base = DbTrainerPredictor.calcular_base_escala([-5.0], [-3.0], [-1.0])
        assert (np.asarray(base) >= 0).all()

    def test_no_altera_el_arreglo_recibido(self):
        # Se la invoca con columnas del dataset de entrenamiento: si mutara la
        # entrada, corromperia ma_3m para el resto del pipeline.
        ma_3m = np.array([0.0, 30.0])
        DbTrainerPredictor.calcular_base_escala(ma_3m, [10.0, 10.0], [20.0, 20.0])
        assert ma_3m == pytest.approx([0.0, 30.0])

    def test_acepta_columnas_de_pandas(self):
        df = pd.DataFrame(
            {"ma_3m": [0.0, 30.0], "lag_1": [10.0, 10.0], "lag_12": [20.0, 20.0]}
        )
        base = DbTrainerPredictor.calcular_base_escala(
            df["ma_3m"], df["lag_1"], df["lag_12"]
        )
        assert base == pytest.approx([10.0, 30.0])


class TestAplicarTechoRelativo:
    """El techo K sobre la prediccion del especialista de ratio.

    Sin techo, un SKU con media movil alta y demanda derrumbada llevaba al
    especialista de ratio a un APE medio de 804%.
    """

    def test_la_constante_k_vale_uno_coma_veinticinco(self):
        assert DbTrainerPredictor.K_TECHO_RELATIVO == 1.25

    def test_respeta_la_prediccion_de_ratio_cuando_esta_por_debajo_del_techo(self):
        combinada = DbTrainerPredictor.aplicar_techo_relativo(
            pred_log=np.array([100.0]), pred_ratio=np.array([110.0])
        )
        assert combinada == pytest.approx([110.0])

    def test_recorta_la_prediccion_de_ratio_cuando_excede_el_techo(self):
        combinada = DbTrainerPredictor.aplicar_techo_relativo(
            pred_log=np.array([100.0]), pred_ratio=np.array([500.0])
        )
        assert combinada == pytest.approx([125.0])

    def test_el_limite_exacto_del_techo_no_se_recorta(self):
        combinada = DbTrainerPredictor.aplicar_techo_relativo(
            pred_log=np.array([100.0]), pred_ratio=np.array([125.0])
        )
        assert combinada == pytest.approx([125.0])

    def test_una_prediccion_logaritmica_nula_no_anula_la_de_ratio(self):
        # El piso de 1.0 sobre pred_log evita que un cero multiplicado por K
        # aplaste a cero toda prediccion de ratio. Deja pasar hasta 1.25 unidades.
        combinada = DbTrainerPredictor.aplicar_techo_relativo(
            pred_log=np.array([0.0]), pred_ratio=np.array([50.0])
        )
        assert combinada == pytest.approx([1.25])

    def test_toma_la_menor_de_las_dos_cuando_el_ratio_sub_predice(self):
        # El hibrido acota por arriba, no por abajo: si el ratio predice menos que
        # el logaritmico, se queda con el ratio.
        combinada = DbTrainerPredictor.aplicar_techo_relativo(
            pred_log=np.array([100.0]), pred_ratio=np.array([40.0])
        )
        assert combinada == pytest.approx([40.0])

    def test_resuelve_el_lote_elemento_por_elemento(self):
        combinada = DbTrainerPredictor.aplicar_techo_relativo(
            pred_log=np.array([100.0, 100.0, 0.0, 8.0]),
            pred_ratio=np.array([110.0, 500.0, 50.0, 4.0]),
        )
        assert combinada == pytest.approx([110.0, 125.0, 1.25, 4.0])

    def test_la_salida_nunca_supera_el_techo(self):
        pred_log = np.linspace(0.0, 1000.0, 200)
        pred_ratio = np.linspace(0.0, 5000.0, 200)
        combinada = DbTrainerPredictor.aplicar_techo_relativo(pred_log, pred_ratio)
        techo = DbTrainerPredictor.K_TECHO_RELATIVO * np.maximum(pred_log, 1.0)
        assert (combinada <= techo + 1e-9).all()

    def test_la_salida_nunca_supera_la_prediccion_de_ratio(self):
        pred_log = np.linspace(0.0, 1000.0, 200)
        pred_ratio = np.linspace(0.0, 5000.0, 200)
        combinada = DbTrainerPredictor.aplicar_techo_relativo(pred_log, pred_ratio)
        assert (combinada <= pred_ratio + 1e-9).all()

    def test_aplicarlo_dos_veces_no_cambia_el_resultado(self):
        # Idempotencia: garantiza que el recorte no se acumula si el hibrido se
        # encadena por error en el flujo de prediccion.
        pred_log = np.array([100.0, 0.0, 8.0])
        una_vez = DbTrainerPredictor.aplicar_techo_relativo(
            pred_log, np.array([500.0, 50.0, 4.0])
        )
        dos_veces = DbTrainerPredictor.aplicar_techo_relativo(pred_log, una_vez)
        assert dos_veces == pytest.approx(una_vez)
