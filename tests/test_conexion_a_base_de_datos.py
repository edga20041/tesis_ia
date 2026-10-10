"""Pruebas de db_connector: armado de la URL de conexion, singleton del motor,
sesiones transaccionales y el chequeo de salud que usa `/health`.

No se conecta a ninguna base real: `create_engine` y el motor se simulan. Lo
que se verifica es la logica de Python alrededor de la conexion (variables de
entorno, URL-encoding de la contraseña, cacheo del Engine, commit/rollback).
"""

import pytest
from unittest.mock import MagicMock

import tesis_ia.utils.db_connector as dbc


@pytest.fixture(autouse=True)
def limpiar_singleton_de_motor():
    """`_engine` es un singleton a nivel de modulo: sin resetearlo, el primer
    test que lo crea contaminaria a todos los que siguen."""
    dbc._engine = None
    yield
    dbc._engine = None


class TestGetDbUrl:
    def test_usa_los_valores_por_defecto_sin_variables_de_entorno(self, monkeypatch):
        for var in ("DB_USER", "DB_PASSWORD", "DB_PASS", "DB_HOST", "DB_PORT", "DB_NAME"):
            monkeypatch.delenv(var, raising=False)

        url = dbc.get_db_url()

        assert url == "mysql+pymysql://root:@localhost:3306/tesis?charset=utf8mb4"

    def test_lee_las_variables_de_entorno_configuradas(self, monkeypatch):
        monkeypatch.setenv("DB_USER", "app_user")
        monkeypatch.setenv("DB_PASSWORD", "secreto")
        monkeypatch.setenv("DB_HOST", "db.interno")
        monkeypatch.setenv("DB_PORT", "3307")
        monkeypatch.setenv("DB_NAME", "produccion")

        url = dbc.get_db_url()

        assert url == "mysql+pymysql://app_user:secreto@db.interno:3307/produccion?charset=utf8mb4"

    def test_usa_db_pass_como_alias_si_falta_db_password(self, monkeypatch):
        monkeypatch.delenv("DB_PASSWORD", raising=False)
        monkeypatch.setenv("DB_PASS", "alias_secreto")

        url = dbc.get_db_url()

        assert "alias_secreto" in url

    def test_codifica_caracteres_especiales_de_la_contrasena(self, monkeypatch):
        monkeypatch.setenv("DB_PASSWORD", "p@ss/word#1")
        monkeypatch.delenv("DB_PASS", raising=False)

        url = dbc.get_db_url()

        assert "p@ss/word#1" not in url
        assert "%40" in url  # '@' codificado


class TestGetEngine:
    def test_crea_el_motor_una_sola_vez(self, monkeypatch):
        motor_falso = MagicMock(name="Engine")
        create_engine_mock = MagicMock(return_value=motor_falso)
        monkeypatch.setattr(dbc, "create_engine", create_engine_mock)

        primero = dbc.get_engine()
        segundo = dbc.get_engine()

        assert primero is segundo is motor_falso
        create_engine_mock.assert_called_once()

    def test_configura_el_pool_con_reciclado_y_pre_ping(self, monkeypatch):
        create_engine_mock = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(dbc, "create_engine", create_engine_mock)

        dbc.get_engine()

        _, kwargs = create_engine_mock.call_args
        assert kwargs["pool_recycle"] == 3600
        assert kwargs["pool_pre_ping"] is True


class TestGetDbSession:
    def test_hace_commit_si_el_bloque_termina_sin_errores(self, monkeypatch):
        sesion = MagicMock(name="Session")
        monkeypatch.setattr(dbc, "get_engine", lambda: MagicMock())
        monkeypatch.setattr(dbc, "sessionmaker", MagicMock(return_value=MagicMock(return_value=sesion)))

        with dbc.get_db_session() as s:
            assert s is sesion

        sesion.commit.assert_called_once()
        sesion.rollback.assert_not_called()
        sesion.close.assert_called_once()

    def test_hace_rollback_y_relanza_si_el_bloque_falla(self, monkeypatch):
        sesion = MagicMock(name="Session")
        monkeypatch.setattr(dbc, "get_engine", lambda: MagicMock())
        monkeypatch.setattr(dbc, "sessionmaker", MagicMock(return_value=MagicMock(return_value=sesion)))

        with pytest.raises(ValueError, match="boom"):
            with dbc.get_db_session() as s:
                raise ValueError("boom")

        sesion.rollback.assert_called_once()
        sesion.commit.assert_not_called()
        sesion.close.assert_called_once()


class TestTestConnection:
    def test_devuelve_true_si_select_1_responde_uno(self, monkeypatch):
        conn = MagicMock()
        conn.execute.return_value.scalar.return_value = 1
        motor = MagicMock()
        motor.connect.return_value.__enter__.return_value = conn
        monkeypatch.setattr(dbc, "get_engine", lambda: motor)

        assert dbc.test_connection() is True

    def test_devuelve_false_si_la_base_no_responde(self, monkeypatch):
        monkeypatch.setattr(dbc, "get_engine", MagicMock(side_effect=RuntimeError("sin conexion")))

        assert dbc.test_connection() is False
