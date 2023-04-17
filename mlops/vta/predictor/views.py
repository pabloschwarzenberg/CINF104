from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views.generic import TemplateView
import requests
import json

from .forms import StudentOutcomeForm

class HomePageView(TemplateView):
    def get(self,request,**kwargs):
        return render(request,"index.html",context=None)
    
def StudentOutcome(request):
    return render(request, "index.html")