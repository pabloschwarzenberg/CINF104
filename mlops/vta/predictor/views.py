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
    if request.method == "POST":
        form = StudentOutcomeForm(request.POST)
        if form.is_valid():
            url="http://localhost:8501/v1/models/reprobacion:predict"
            solved=[]
            if form.cleaned_data["e0"]:
                solved.append(1)
            else:
                solved.append(0)
            if form.cleaned_data["e1"]:
                solved.append(1)
            else:
                solved.append(0)
            if form.cleaned_data["e42"]:
                solved.append(1)
            else:
                solved.append(0)
            data={"instances":[solved]}
            header={"Content-Type":"application/json"}
            result=requests.post(url,data=json.dumps(data),headers=header)
            if result.status_code == 200:
                answer=result.json()
                print(solved)
                print(answer)
                if answer["predictions"][0][0] > answer["predictions"][0][1]:
                    outcome="Pass"
                else:
                    outcome="Fail"
            else:
                outcome="Unknown"
            return render(request,"predicted_outcome.html", {"outcome": outcome,"solved":solved})
    else:
        form = StudentOutcomeForm()

    return render(request, "student_outcome.html", {"form": form})