from django.urls import path
from . import views

app_name = 'ar_visualization'

urlpatterns = [
    path('ar/', views.ar_page, name='ar_page'),
    path('get_crop_model/<str:crop_type>/', views.get_crop_model, name='get_crop_model'),
    path('get_crop_labels/<str:crop_type>/', views.get_crop_labels, name='get_crop_labels'),
    path('save_crop_labels/<str:crop_type>/', views.save_crop_labels, name='save_crop_labels'),
    path('camera/', views.ar_camera_view, name='ar_camera'),
    path('scan/', views.scan_page, name='scan_page'),
]