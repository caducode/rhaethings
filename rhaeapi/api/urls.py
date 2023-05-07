from django.urls import path
from . import views

# endpoints
urlpatterns = [
    path('fotos/similar/', views.getSimilar),
]