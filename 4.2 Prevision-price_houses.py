import pandas as pd
base = pd.read_csv('house_prices.csv')

#apresenta todos os dados dentro da planilha
print(base.head())
#apresenta a quantidade de cada dado dentro da planilha
print(base.count())
#demostra quantas linhas e colunas tem os dados importados
print(base.shape)

#O Iloc chama a coluna de dados que quer, e values é a tranformação em np.array
x = base.iloc[:, 5].values
x = x.reshape(-1, 1)
print(x.shape)

#Colocando dois pontos nas colunas, e já com values, não precisa do reshape
y = base.iloc[:, 2:3].values
print(y.shape)

from sklearn.preprocessing import StandardScaler 
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

print(x, y)


import matplotlib.pyplot as plt
plt.scatter(x,y)

import numpy as np
np.random.seed(1)
np.random.rand(2)

import tensorflow as tf

#células iniciais
b0 = tf.Variable(0.41)
b1 = tf.Variable(0.72)
#cria uma quantidade de dados que será utilizada por aprendizagem, para não pesar 
batch_size = 32
# Em cada plchr de dados então será utilizado um qtd de 32 dados por aprendizagem
xph = tf.placeholder(tf.float32, [batch_size,1])
yph = tf.placeholder(tf.float32,[batch_size, 1])

# Define as vairáveis
y_modelo  = b0 + b1*xph
erro = tf.losses.mean_squared_error(yph,y_modelo)
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)
init = tf.global_variables_initializer()

#Abre a sessão com aprendiado de 1000 vezes
with tf.Session() as ses:
    ses.run(init)
    for i in range(10000):
        # Esta função envia no intervalo da qtd de x, 32 dados aleatórios
        indices = np.random.randint(len(x), size = batch_size)
        # O dicionário que alimentará os plchlr
        feed = {xph: x[indices], yph: y[indices]}
        #Roda a aprendizagem de máquina com o dicionário estabelecido
        ses.run(treinamento, feed_dict = feed)
    b0_final, b1_final = ses.run([b0, b1])
print(b0_final, b1_final)

# Valores etimados das casas pela nossa regressão
previsoes = b0_final + b1_final * x
plt.plot(x,  previsoes, color = 'red')
# Isso reverte o escalonamento das variáveis 
y1 = scaler_y.inverse_transform(y)
previsoes1 = scaler_y.inverse_transform(previsoes)
print(y1)
# Define quantos doláres o estimador dá de erro para mais e menos
from sklearn.metrics import mean_absolute_error
mae= mean_absolute_error(y1, previsoes1)
# O que no caso resulta em 173 mil doláres a mais ou menos do valor real
print(mae)
