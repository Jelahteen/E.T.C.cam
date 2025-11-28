from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login_page'),
    path('signup/', views.signup_view, name='signup_page'),
    path('dashboard/', views.dashboard_view, name='dashboard_page'),
    path('list/', views.list_page_view, name='list_page'),  # Fixed: list_page_view instead of list_view
    path('', views.homepage, name='homepage'),  # Fixed: removed extra slash
]