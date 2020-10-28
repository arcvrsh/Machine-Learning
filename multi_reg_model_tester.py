import pickle as pkl
import numpy as np


def load_model(path):
    with open(path,'rb') as f:
       return pkl.load(f)

encoder = load_model('models/state_hot_encoder.pkl')
model = load_model('models/startup_profit_prediction.pkl')

rnd = int(input('enter R&D spend: '))
admin = int(input('enter admin spend: '))
mkt = int(input('enter marketing spend: '))
state = input('select state (California,Florida,NewYork): ')

state_dummies = encoder.transform([[state]])
data = np.array([[rnd,admin,mkt]])
input_data = np.append(data,state_dummies.toarray())
profit = model.predict(input_data.reshape(1,-1))