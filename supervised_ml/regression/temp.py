# basic imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# algo and and estimators
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

trainx, testx, trainy, testy = train_test_split(x,y,test_size=.2,random_state=0)

def save_model(path,model):
    with open(path,'wb') as f:
        pkl.dump(model,f)
    return path

def load_model(path):
    with open(path,'rb') as f:
       return pkl.load(f)