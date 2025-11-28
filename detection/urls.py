from django.urls import path
from . import views

app_name = 'detection'  # Defines the namespace for these URLs

urlpatterns = [
    # Path for rendering the initial scan page (GET request)
    path('scan/', views.scan_view, name='scan_page'),
    # Path for handling the image detection POST request
    # Use the same URL pattern that matches your JavaScript request
    path('detect/', views.detect_view, name='detect'),
    path('debug-db/', views.debug_database, name='debug_db'),
]