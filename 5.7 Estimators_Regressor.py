import pandas as pd
import tensorflow as tf
import numpy as np

base = pd.read_csv('house_prices.csv')
#print(base.columns) - pega os dados numéricos para fazer a regressão
colunas_usadas = [ 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']
base = pd.read_csv('house_prices.csv', usecols =colunas_usadas)
from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']] = scaler_x.fit_transform(base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']])
scaler_y = MinMaxScaler()
base[['price']] = scaler_y.fit_transform(base[['price']]) # Padroniza os preços das casas
x = base.drop('price', axis =1) # define todas as colunas menos a price 
y = base.price
previsores_colunas = colunas_usadas[1:17]

colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas] #Vai de coluna em coluna padroniza
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)

func_train = tf.estimator.inputs.pandas_input_fn(x = x_train,y = y_train, batch_size= 32
                                                 , num_epochs = None, shuffle = True)
regressor = tf.estimator.DNNRegressor(hidden_units =[8,8,8], feature_columns=colunas)
regressor.train(input_fn = func_train, steps = 20000)

func_predict =  tf.estimator.inputs.pandas_input_fn(x = x_test, shuffle =False) #shuffle false deixa a ordem original
predict = regressor.predict(input_fn=func_predict)

values_predict =[ ]
for p in regressor.predict(input_fn = func_predict):
    values_predict.append(p['predictions'][0])
    
values_predict = np.asarray(values_predict).reshape(-1,1)
values_predict = scaler_y.inverse_transform(values_predict)

y_test2 = y_test.reshape(-1,1)
y_test2 - scaler_y.inverse_transform(y_test2)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test2, values_predict)