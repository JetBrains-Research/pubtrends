"""
Flask configuration (class-based, environment-driven).

Usage:
  app.config.from_object('pysrc.app.config.Config')

Environment variables (examples):
  - FLASK_ENV or APP_ENV: development|testing|production (default: production)
  - SECRET_KEY
  - DATABASE_FILE (default: db.sqlite)
  - SQLALCHEMY_ECHO (true/false)
  - SQLALCHEMY_TRACK_MODIFICATIONS (true/false, default: false)
  - SECURITY_URL_PREFIX (default: /admin)
  - SECURITY_PASSWORD_HASH (default: pbkdf2_sha512)
  - SECURITY_PASSWORD_SALT
  - SECURITY_LOGIN_URL (default: /login/)
  - SECURITY_LOGOUT_URL (default: /logout/)
  - SECURITY_POST_LOGIN_VIEW (default: /admin/)
  - SECURITY_POST_LOGOUT_VIEW (default: /)
  - SECURITY_REGISTERABLE (true/false, default: false)
  - SECURITY_SEND_REGISTER_EMAIL (true/false, default: false)
"""

from __future__ import annotations

import os

# Optional: load variables from a .env file if python-dotenv is available.
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:  # pragma: no cover - silently ignore
    pass


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_str(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


class BaseConfig:
    # Core
    SECRET_KEY = _env_str("SECRET_KEY") or os.urandom(24).hex()

    # Database
    DATABASE_FILE = _env_str("DATABASE_FILE", "db.sqlite")
    # SQLAlchemy URI is constructed by application code based on DATABASE_FILE
    SQLALCHEMY_ECHO = _env_bool("SQLALCHEMY_ECHO", False)
    SQLALCHEMY_TRACK_MODIFICATIONS = _env_bool("SQLALCHEMY_TRACK_MODIFICATIONS", False)

    # Flask-Security
    SECURITY_URL_PREFIX = _env_str("SECURITY_URL_PREFIX", "/admin")
    SECURITY_PASSWORD_HASH = _env_str("SECURITY_PASSWORD_HASH", "pbkdf2_sha512")
    # Keep previous default for salt if env not provided
    SECURITY_PASSWORD_SALT = _env_str("SECURITY_PASSWORD_SALT", "D82CWwxNDzdhB7mbeaAChVd2BjdM9VjR")

    # Flask-Security URLs
    SECURITY_LOGIN_URL = _env_str("SECURITY_LOGIN_URL", "/login/")
    SECURITY_LOGOUT_URL = _env_str("SECURITY_LOGOUT_URL", "/logout/")
    # SECURITY_REGISTER_URL intentionally left optional

    SECURITY_POST_LOGIN_VIEW = _env_str("SECURITY_POST_LOGIN_VIEW", "/admin/")
    SECURITY_POST_LOGOUT_VIEW = _env_str("SECURITY_POST_LOGOUT_VIEW", "/")
    # SECURITY_POST_REGISTER_VIEW optional

    # Features
    SECURITY_REGISTERABLE = _env_bool("SECURITY_REGISTERABLE", False)
    SECURITY_SEND_REGISTER_EMAIL = _env_bool("SECURITY_SEND_REGISTER_EMAIL", False)

    # Flask flags (commonly toggled by environment)
    DEBUG = _env_bool("DEBUG", False)
    TESTING = _env_bool("TESTING", False)


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class TestingConfig(BaseConfig):
    TESTING = True
    # Use a separate DB file for tests unless explicitly provided
    DATABASE_FILE = _env_str("DATABASE_FILE", "test_db.sqlite")


class ProductionConfig(BaseConfig):
    DEBUG = False
    TESTING = False


def _select_config_class() -> type[BaseConfig]:
    env = (os.environ.get("APP_ENV") or os.environ.get("FLASK_ENV") or "production").lower()
    if env.startswith("dev"):
        return DevelopmentConfig
    if env.startswith("test"):
        return TestingConfig
    return ProductionConfig


# Export a chosen configuration class for `from_object` usage.
Config: type[BaseConfig] = _select_config_class()
