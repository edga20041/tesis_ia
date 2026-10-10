"""Pruebas de ExogenousPipeline: clima, calendario, macroeconomia y su sincronizacion a MySQL.

fetch_calendar_month y fetch_macro_month son formulas puras (no tocan la red ni
la base) y se prueban directamente. fetch_weather_month llama a la API de
Open-Meteo via requests: se simula la respuesta HTTP. sync_month/sync_range
escriben en MySQL: se simula el motor, igual que en los demas modulos de este
paquete.
"""

import datetime

import pytest
from unittest.mock import MagicMock

from tesis_ia.data.exogenous_pipeline import ExogenousPipeline
import tesis_ia.data.exogenous_pipeline as ep


@pytest.fixture
def pipeline():
    return ExogenousPipeline(engine=MagicMock(name="engine"))


class TestFetchCalendarMonth:
    def test_marca_diciembre_como_festivo_comercial_y_con_aguinaldo(self, pipeline):
        resultado = pipeline.fetch_calendar_month(2025, 12)
        assert resultado["tiene_festivo_comercial"] == 1
        assert resultado["quincena_aguinaldo"] == 1
        assert resultado["inicio_ciclo_escolar"] == 0

    def test_marca_febrero_como_inicio_de_ciclo_escolar_sin_festivo_comercial(self, pipeline):
        resultado = pipeline.fetch_calendar_month(2025, 2)
        assert resultado["inicio_ciclo_escolar"] == 1
        assert resultado["tiene_festivo_comercial"] == 0
        assert resultado["quincena_aguinaldo"] == 0

    def test_mayo_y_junio_son_festivo_comercial_por_dia_de_la_madre_y_del_padre(self, pipeline):
        assert pipeline.fetch_calendar_month(2025, 5)["tiene_festivo_comercial"] == 1
        assert pipeline.fetch_calendar_month(2025, 6)["tiene_festivo_comercial"] == 1

    def test_un_mes_sin_marcas_especiales_no_activa_ninguna_bandera(self, pipeline):
        resultado = pipeline.fetch_calendar_month(2025, 9)
        assert resultado["tiene_festivo_comercial"] == 0
        assert resultado["quincena_aguinaldo"] == 0
        assert resultado["inicio_ciclo_escolar"] == 0

    def test_cuenta_los_feriados_nacionales_reales_de_paraguay(self, pipeline):
        resultado = pipeline.fetch_calendar_month(2025, 5)
        assert resultado["cantidad_feriados"] >= 2
        assert resultado["nombre_festivo"] is not None

    def test_devuelve_nombre_festivo_nulo_si_no_hay_feriados_en_el_mes(self, pipeline):
        resultado = pipeline.fetch_calendar_month(2025, 9)
        if resultado["cantidad_feriados"] == 0:
            assert resultado["nombre_festivo"] is None

    def test_el_ciclo_de_cobro_mensual_esta_siempre_activo(self, pipeline):
        for mes in range(1, 13):
            assert pipeline.fetch_calendar_month(2025, mes)["ciclo_cobro_mensual"] == 1


class TestFetchMacroMonth:
    def test_el_ipc_parte_de_cien_en_enero_2021(self, pipeline):
        resultado = pipeline.fetch_macro_month(2021, 1)
        assert resultado["ipc_indice"] == pytest.approx(100.0)

    def test_el_ipc_crece_de_forma_compuesta_mes_a_mes(self, pipeline):
        enero = pipeline.fetch_macro_month(2022, 1)["ipc_indice"]
        febrero = pipeline.fetch_macro_month(2022, 2)["ipc_indice"]
        assert febrero > enero

    def test_el_tipo_de_cambio_aumenta_con_el_anio_y_el_mes(self, pipeline):
        base = pipeline.fetch_macro_month(2021, 1)["tipo_cambio_pyg_usd"]
        un_anio_despues = pipeline.fetch_macro_month(2022, 1)["tipo_cambio_pyg_usd"]
        assert un_anio_despues == pytest.approx(base + 150)

    def test_devuelve_las_siete_variables_macro_esperadas(self, pipeline):
        resultado = pipeline.fetch_macro_month(2025, 6)
        assert set(resultado) == {
            "ipc_indice", "ipc_var_interanual", "tipo_cambio_pyg_usd",
            "pbi_crecimiento", "tasa_desempleo", "remesas_usd", "inversion_ext_usd",
        }


class TestFetchWeatherMonth:
    def _respuesta_falsa(self, monkeypatch, payload, status_ok=True):
        resp = MagicMock()
        resp.json.return_value = payload
        if not status_ok:
            resp.raise_for_status.side_effect = Exception("HTTP 500")
        monkeypatch.setattr(ep.requests, "get", MagicMock(return_value=resp))
        return resp

    def test_promedia_temperaturas_y_suma_precipitacion_del_mes(self, pipeline, monkeypatch):
        self._respuesta_falsa(monkeypatch, {
            "daily": {
                "temperature_2m_max": [30.0, 32.0, 28.0],
                "temperature_2m_mean": [25.0, 27.0, 23.0],
                "precipitation_sum": [0.0, 5.0, 10.0],
            }
        })

        resultado = pipeline.fetch_weather_month(2025, 1)

        assert resultado["temp_max_promedio"] == pytest.approx(30.0)
        assert resultado["temp_media_promedio"] == pytest.approx(25.0)
        assert resultado["precipitacion_mm"] == pytest.approx(15.0)

    def test_usa_el_ultimo_dia_de_diciembre_como_fin_de_rango(self, pipeline, monkeypatch):
        resp = self._respuesta_falsa(monkeypatch, {"daily": {}})
        pipeline.fetch_weather_month(2025, 12)
        params = ep.requests.get.call_args.kwargs["params"]
        assert params["end_date"] == "2025-12-31"

    def test_devuelve_valores_por_defecto_si_la_api_falla(self, pipeline, monkeypatch):
        self._respuesta_falsa(monkeypatch, {}, status_ok=False)

        resultado = pipeline.fetch_weather_month(2025, 1)

        assert resultado == {
            "temp_max_promedio": 30.0,
            "temp_media_promedio": 25.0,
            "precipitacion_mm": 100.0,
        }

    def test_devuelve_valores_por_defecto_si_la_api_no_trae_datos_diarios(self, pipeline, monkeypatch):
        self._respuesta_falsa(monkeypatch, {"daily": {"temperature_2m_max": []}})

        resultado = pipeline.fetch_weather_month(2025, 1)

        assert resultado["temp_max_promedio"] == 30.0


class TestSyncMonth:
    def _mockear_fetchers(self, monkeypatch, pipeline):
        monkeypatch.setattr(pipeline, "fetch_weather_month", lambda a, m: {
            "temp_max_promedio": 30.0, "temp_media_promedio": 25.0, "precipitacion_mm": 100.0,
        })
        monkeypatch.setattr(pipeline, "fetch_calendar_month", lambda a, m: {
            "cantidad_feriados": 1, "tiene_festivo_comercial": 0, "nombre_festivo": "Feriado X",
            "quincena_aguinaldo": 0, "inicio_ciclo_escolar": 0, "ciclo_cobro_mensual": 1,
        })
        monkeypatch.setattr(pipeline, "fetch_macro_month", lambda a, m: {
            "ipc_indice": 101.0, "ipc_var_interanual": 4.5, "tipo_cambio_pyg_usd": 7200.0,
            "pbi_crecimiento": 3.8, "tasa_desempleo": 5.2, "remesas_usd": 5e7, "inversion_ext_usd": 2.5e7,
        })

    def test_inserta_las_tres_tablas_y_devuelve_los_detalles(self, monkeypatch, pipeline):
        self._mockear_fetchers(monkeypatch, pipeline)
        conn = MagicMock()
        pipeline.engine.begin.return_value.__enter__.return_value = conn

        ok, details = pipeline.sync_month(2025, 6)

        assert ok is True
        assert conn.execute.call_count == 3
        assert details["calendario"]["nombre_festivo"] == "Feriado X"
        assert details["macro"]["ipc_indice"] == 101.0

    def test_devuelve_false_si_falla_la_escritura_pero_conserva_los_detalles(self, monkeypatch, pipeline):
        self._mockear_fetchers(monkeypatch, pipeline)
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("conexion perdida")
        pipeline.engine.begin.return_value.__enter__.return_value = conn

        ok, details = pipeline.sync_month(2025, 6)

        assert ok is False
        assert "clima" in details


class TestSyncRange:
    def test_recorre_el_rango_cruzando_el_fin_de_anio(self, monkeypatch, pipeline):
        llamadas = []
        monkeypatch.setattr(pipeline, "sync_month", lambda a, m: (llamadas.append((a, m)), (True, {}))[1])

        total = pipeline.sync_range(2024, 11, 2025, 2)

        assert llamadas == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]
        assert total == 4

    def test_no_cuenta_los_meses_en_que_la_sincronizacion_falla(self, monkeypatch, pipeline):
        resultados = iter([(True, {}), (False, {}), (True, {})])
        monkeypatch.setattr(pipeline, "sync_month", lambda a, m: next(resultados))

        total = pipeline.sync_range(2025, 1, 2025, 3)

        assert total == 2
