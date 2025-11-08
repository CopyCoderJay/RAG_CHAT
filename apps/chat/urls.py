from django.urls import path
from . import views

app_name = 'chat'
urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('<str:chat_id>/', views.chat_detail_view, name='detail'),
]

