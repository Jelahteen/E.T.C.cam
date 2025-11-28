from django.contrib import admin
from django.urls import path, include
from core import views as core_views
from django.conf import settings
from django.conf.urls.static import static

# --- IMPORTS FOR INTERNATIONALIZATION ---
from django.conf.urls.i18n import i18n_patterns
from django.views.i18n import set_language
# --- END INTERNATIONALIZATION IMPORTS ---

# These URLs are not language-prefixed
urlpatterns = [
    # This URL is for handling language switching via a POST request
    path('i18n/setlang/', set_language, name='set_language'),
    # API endpoint for scan history - ADD THIS LINE
    path('api/scan-history/', core_views.scan_history_api, name='scan_history_api'),  # ✅ This is already correct!
]

# All the URLs inside i18n_patterns will be automatically prefixed with the active language code
# e.g., /en/dashboard/, /tl/dashboard/
urlpatterns += i18n_patterns(
    path('admin/', admin.site.urls),  # The admin site is also language-prefixed
    path('detection/', include('detection.urls')),
    path('ar_visualization/', include('ar_visualization.urls')),
    path('', core_views.homepage, name='homepage'),
    path('login/', core_views.login_view, name='login_page'),
    path('signup/', core_views.signup_view, name='signup_page'),
    path('dashboard/', core_views.dashboard_view, name='dashboard_page'),
    path('logout/', core_views.logout_view, name='logout_view'),
    path('list/', core_views.list_page_view, name='list_page'),
)

# Serves static files AND media files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)