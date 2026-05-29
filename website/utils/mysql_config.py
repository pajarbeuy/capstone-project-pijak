import os
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    database: str
    user: str
    password: str


def _secret_value(section: str, key: str, default: str | None = None) -> str | None:
    try:
        return st.secrets.get(section, {}).get(key, default)
    except Exception:
        return default


def load_mysql_config() -> MySQLConfig:
    return MySQLConfig(
        host=os.getenv("MYSQL_HOST") or _secret_value("mysql", "host", "localhost"),
        port=int(os.getenv("MYSQL_PORT") or _secret_value("mysql", "port", "3306")),
        database=os.getenv("MYSQL_DATABASE") or _secret_value("mysql", "database", "capstone_pijak"),
        user=os.getenv("MYSQL_USER") or _secret_value("mysql", "user", "root"),
        password=os.getenv("MYSQL_PASSWORD") or _secret_value("mysql", "password", ""),
    )


def mysql_connection_kwargs() -> dict[str, str | int]:
    config = load_mysql_config()
    return {
        "host": config.host,
        "port": config.port,
        "database": config.database,
        "user": config.user,
        "password": config.password,
    }


def get_mysql_connection():
    import mysql.connector

    return mysql.connector.connect(**mysql_connection_kwargs())
