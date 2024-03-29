from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from . import views


app_name = "retinaface_detection"

urlpatterns = [
    # two paths: with or without given image
    path("", views.upload_file, name="upload_file"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
