"""Pruebas de FeatureCalculator: consolidacion semanal->mensual y calculo de
lags/media movil/market share/price gap sobre ventas_mensuales en MySQL.

Los tres metodos son UPDATE/INSERT puros contra la base; no hay logica en
Python mas alla de pasar los parametros correctos y manejar la excepcion. Se
simula el motor para verificar exactamente eso: que la consulta se ejecuta con
los parametros esperados y que una falla de la base no hace caer el proceso.
"""

import pytest
from unittest.mock import MagicMock

from tesis_ia.data.feature_calculator import FeatureCalculator


@pytest.fixture
def calculator():
    return FeatureCalculator(engine=MagicMock(name="engine"))


class TestConsolidateWeeklySales:
    def test_devuelve_la_cantidad_de_filas_afectadas(self, calculator):
        conn = MagicMock()
        conn.execute.return_value.rowcount = 42
        calculator.engine.begin.return_value.__enter__.return_value = conn

        resultado = calculator.consolidate_weekly_sales(2025, 3)

        assert resultado == 42
        params = conn.execute.call_args.args[1]
        assert params == {"anio": 2025, "mes": 3}

    def test_devuelve_cero_si_falla_la_consolidacion(self, calculator):
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("deadlock")
        calculator.engine.begin.return_value.__enter__.return_value = conn

        resultado = calculator.consolidate_weekly_sales(2025, 3)

        assert resultado == 0


class TestComputeFeaturesMonth:
    def test_ejecuta_las_tres_actualizaciones_en_una_sola_transaccion(self, calculator):
        conn = MagicMock()
        calculator.engine.begin.return_value.__enter__.return_value = conn

        ok = calculator.compute_features_month(2025, 7)

        assert ok is True
        assert conn.execute.call_count == 3
        for llamada in conn.execute.call_args_list:
            assert llamada.args[1] == {"anio": 2025, "mes": 7}

    def test_devuelve_false_si_alguna_actualizacion_falla(self, calculator):
        conn = MagicMock()
        conn.execute.side_effect = [None, RuntimeError("timeout"), None]
        calculator.engine.begin.return_value.__enter__.return_value = conn

        ok = calculator.compute_features_month(2025, 7)

        assert ok is False
