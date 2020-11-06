import pickle as pkl
import numpy as np

path = 'models/slr_hpvsprice_predictor.pkl'
with open(path,'rb') as f:
    model = pkl.load(f)

hp = int(input('enter horsepower of vehicle'))
# convert single value into 2d array
x = np.array([[hp]])
ypred = model.predict(x)
print(f'hp = {hp} \n => price = {ypred[0]}')