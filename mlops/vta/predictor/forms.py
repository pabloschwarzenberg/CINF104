from django import forms

class StudentOutcomeForm(forms.Form):
    e0 = forms.BooleanField(label="e0",required=False,widget=forms.CheckboxInput(attrs={"class":"form-check-input"}))
    e1 = forms.BooleanField(label="e1",required=False,widget=forms.CheckboxInput(attrs={"class":"form-check-input"}))
    e42 = forms.BooleanField(label="e42",required=False,widget=forms.CheckboxInput(attrs={"class":"form-check-input"}))