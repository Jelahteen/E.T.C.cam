# etccam_project/settings.py
"""
Django settings for etccam_project project.
... (rest of your settings file)
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-32fcd6t$n*6%ub*_qjgd*!7(^miqa=fft1lg&c5+=m7d26$qql'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'core',
    'detection',
    'ar_visualization',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    # --- ADD THIS LINE FOR INTERNATIONALIZATION ---
    'django.middleware.locale.LocaleMiddleware', 
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'etccam_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'core', 'templates')], # Assuming your templates are here
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                # --- ADD THIS LINE FOR INTERNATIONALIZATION ---
                'django.template.context_processors.i18n', 
            ],
        },
    },
]

WSGI_APPLICATION = 'etccam_project.wsgi.application'

# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'etccam_db',
        'USER': 'jelah',
        'PASSWORD': 'qwertyuiop1019',
        'HOST': '127.0.0.1',
        'PORT': '3306',
        'OPTIONS': {
            'charset': 'utf8mb4',
        },
    }
}

# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.docs.djangoproject.com/en/5.2/topics/i18n/
LANGUAGE_CODE = 'en-us' # Default language for the project

TIME_ZONE = 'UTC'

USE_I18N = True # Keep this True for general Django use

USE_TZ = True

# --- ADD THESE LINES FOR INTERNATIONALIZATION ---
LANGUAGES = [
    ('en', 'English'),
    ('tl', 'Filipino'),
]

# Path where Django will look for translation files (relative to BASE_DIR)
LOCALE_PATHS = [
    os.path.join(BASE_DIR, 'locale'),
]
# --- END INTERNATIONALIZATION ADDITIONS ---

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static_root') # Ensure this path is outside your app's static files
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'), # Your project's general static files (e.g., img, css, js)
    os.path.join(BASE_DIR, 'ar_visualization', 'static'), # If ar_visualization has its own static folder
]

# --- ADD THESE LINES FOR MEDIA FILES (Uploaded images) ---
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
# --- END MEDIA FILES ADDITIONS ---

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# New login and logout redirects
LOGIN_REDIRECT_URL = 'dashboard_page'
LOGIN_URL = 'login_page'
LOGOUT_REDIRECT_URL = 'homepage'