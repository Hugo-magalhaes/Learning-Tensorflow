import pandas as pd
base = pd.read_csv('house_prices.csv')
print(base.columns)
colunas_usadas =   [ 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']
base = pd.read_csv('house_prices.csv', usecols = colunas_usadas)
from sklearn.preprocessing import MinMaxScaler

# Escalona os dados de x
scaler_x = MinMaxScaler()
base[[ 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']] = scaler_x.fit_transform(base[[ 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']])
scaler_y  = MinMaxScaler()

# Escalona os valores de y
base[['price']] = scaler_y.fit_transform(base[['price']])
X = base.drop('price', axis = 1)
Y = base.price
previsoes_colunas = colunas_usadas[1:17]

# define a quantidde de variaveis para treinamento e teste 
import tensorflow as tf
colunas = [tf.feature_column.numeric_column(key = c ) for c in previsoes_colunas]
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, Y_treinamento , Y_teste = train_test_split(X, Y, test_size = 0.3)

# Cria a função que vai treinar a regressão e faz o teste dela
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = X_treinamento, y = Y_treinamento,
                                                         batch_size = 32, num_epochs = None,
                                                         shuffle = True)

funcao_teste = tf.estimator.inputs.pandas_input_fn(x = X_teste, y = Y_teste,
                                                         batch_size= 32, num_epochs = 10000,
                                                         shuffle = False)

# Aqui avalia quanto o software aprendeu e as previsões que ele pode dar
regressor = tf.estimator.LinearRegressor(feature_columns = colunas)
regressor.train(input_fn = funcao_treinamento, steps = 10000)

metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento, steps = 10000)
metricas_teste = regressor.evaluate(input_fn = funcao_teste, steps = 10000)

funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = X_teste, shuffle = None)
previsoes = regressor.predict(input_fn = funcao_previsao)

# Transformou array em matrizes para conversão e ánalise de erro
valores_previsoes = []
for p in regressor.predict(input_fn = funcao_previsao):
    valores_previsoes.append(p['predictions'])
import numpy as np

valores_previsoes = np.array(valores_previsoes).reshape(-1,1)
valores_previsoes = scaler_y.inverse_transform(valores_previsoes)

Y_teste2 = Y_teste.values.reshape(-1,1)
Y_teste2 = scaler_y.inverse_transform(Y_teste2)

from sklearn.metrics import mean_absolute_error as er
mae = er(Y_teste2,valores_previsoes)
print(mae)