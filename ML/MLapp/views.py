from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from sklearn import linear_model

# Create your views here.
def index(request):
    return render(request,'index.html')

def results(request):
    df = pd.read_csv('homeprices.csv')
    new_df = df.drop('price',axis='columns')
    price = df.price
    reg = linear_model.LinearRegression()
    reg.fit(new_df,price)
    msg = int(request.GET['message'])
    pred = reg.predict([[msg]])
    Answer = round(pred[0],2)
    return render(request,'result.html',{"Answer":Answer})
