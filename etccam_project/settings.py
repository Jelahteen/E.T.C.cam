# etccam_project/settings.py

import os
from pathlib import Path
from django.core.management.utils import get_random_secret_key
import dj_database_url  # MOVED TO TOP - REQUIRED FOR RENDER POSTGRESQL

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Security / Debug
# -----------------------------------------------------------------------------

# Read SECRET_KEY from env in production; fall back to your existing key for dev
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-32fcd6t$n*6%ub*_qjgd*!7(^miqa=fft1lg&c5+=m7d26$qql",
)

# DEBUG: False on Render, True locally
DEBUG = os.environ.get("DJANGO_DEBUG", "False").lower() == "true"

# Hosts – include Render host automatically if available
ALLOWED_HOSTS = []

RENDER_EXTERNAL_HOSTNAME = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
if RENDER_EXTERNAL_HOSTNAME:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)

# Optional: allow localhost/127.0.0.1 for development
ALLOWED_HOSTS += ["localhost", "127.0.0.1", "e-t-c-cam.onrender.com"]

# -----------------------------------------------------------------------------
# Application definition
# -----------------------------------------------------------------------------

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "core",
    "detection",
    "ar_visualization",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    # Internationalization
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "etccam_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "core", "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                # Internationalization
                "django.template.context_processors.i18n",
            ],
        },
    },
]

WSGI_APPLICATION = "etccam_project.wsgi.application"

# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
# Development: your Laragon MySQL (local)
# Production: use DATABASE_URL (Render PostgreSQL)
# -----------------------------------------------------------------------------

# Flag to detect if DATABASE_URL is configured (for Render / production)
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    # PRODUCTION: Render PostgreSQL (ssl_require=True REQUIRED)
    DATABASES = {
        "default": dj_database_url.config(
            default=DATABASE_URL,
            conn_max_age=600,
            ssl_require=True,  # FIXED: Render PostgreSQL REQUIRES SSL
        )
    }
else:
    # LOCAL DEVELOPMENT: Your Laragon MySQL
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": "etccam_db",
            "USER": "root",
            "PASSWORD": "",
            "HOST": "127.0.0.1",
            "PORT": "3306",
            "OPTIONS": {
                "charset": "utf8mb4",
            },
        }
    }

# -----------------------------------------------------------------------------
# Password validation
# -----------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# -----------------------------------------------------------------------------
# Internationalization
# -----------------------------------------------------------------------------

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

LANGUAGES = [
    ("en", "English"),
    ("tl", "Filipino"),
]

LOCALE_PATHS = [
    os.path.join(BASE_DIR, "locale"),
]

# -----------------------------------------------------------------------------
# Static & Media files
# -----------------------------------------------------------------------------

STATIC_URL = "static/"

# Folder where collectstatic will put compiled static files (for Render)
STATIC_ROOT = os.path.join(BASE_DIR, "static_root")

# Additional static folders (development & app-level)
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"),
    os.path.join(BASE_DIR, "ar_visualization", "static"),
]

MEDIA_URL = "/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, "media")

# -----------------------------------------------------------------------------
# Default primary key field type
# -----------------------------------------------------------------------------

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# -----------------------------------------------------------------------------
# Auth redirects
# -----------------------------------------------------------------------------

LOGIN_REDIRECT_URL = "dashboard_page"
LOGIN_URL = "login_page"
LOGOUT_REDIRECT_URL = "homepage"