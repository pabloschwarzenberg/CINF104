from django.urls import path
from predictor import views

urlpatterns = [
    path("",views.HomePageView.as_view(),name="index"),
    path("student_outcome/",views.StudentOutcome,name="student_outcome")
]