import pandas as pd
base = pd.read_csv('house_prices.csv')
base.head()


x = base.iloc[:,5:6].values
y =base.iloc[:,2:3].values

#Linearizando a base de dados
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
Y = scaler_y.fit_transform(y)

# Regressão dos dados para seguir uma linha reta
import tensorflow as tf
colunas = [tf.feature_column.numeric_column('x', shape = [1])]
regressor = tf.estimator.LinearRegressor(feature_columns=colunas)
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, Y_treinamento , Y_teste = train_test_split(X, Y, test_size = 0.3)
print(Y_treinamento.shape, Y_teste.shape) 

# Funções para treinar e parametrizar os valores, assim tendo um set de onde reverter
funcao_treinamento = tf.estimator.inputs.numpy_input_fn({'x' : X_treinamento}, Y_treinamento, batch_size = 32, num_epochs= None, shuffle = True )
funcao_teste = tf.estimator.inputs.numpy_input_fn({'x':X_teste}, Y_teste, batch_size =32, num_epochs = 1000, shuffle = False)
regressor.train(input_fn = funcao_treinamento, steps = 10000)
metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento, steps = 10000)
metricas_teste = regressor.evaluate(input_fn = funcao_teste, steps = 10000)

# Dados de teste para estimar e reverter os valores para doláres
import numpy as np
novas_casas = np.array([[800],[900],[1000]])
novas_casas = scaler_x.transform(novas_casas)
funcao_previsao = tf.estimator.inputs.numpy_input_fn({'x': novas_casas}, shuffle = False)
previsoes = regressor.predict(input_fn = funcao_previsao)
for p in regressor.predict(input_fn = funcao_previsao):
    print(p['predictions'])
    print(scaler_y.inverse_transform(p['predictions']))