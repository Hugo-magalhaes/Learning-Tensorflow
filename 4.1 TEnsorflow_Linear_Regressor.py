import numpy as np 
x = np.array([[18],[23],[28],[33],[38],[43],[48],[53], [58],[63]])
y = np.array([[871],[1132],[1042],[1356],[1488],[1638],[1569],[1754],[1866],[1900]])


from sklearn.preprocessing import StandardScaler 
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
#print(x)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)
#print('\n', y)

import matplotlib.pyplot as plt

#Formula regressão linear simples
'y = b0 + b1*x'
#Observar valores aleatórios para se inializar a função
np.random.seed(0)
np.random.rand(2)

import tensorflow as tf
b0 = tf.Variable(0.54)
b1 = tf.Variable(0.71)


#criou-se as variaveis de erro, otimização e o que treina o códgio a minizar o erro
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
tf1.disable_eager_execution()

erro = tf.losses.mean_squared_error(y,(b0+b1*x))
otimizador = tf1.train.GradientDescentOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)
init = tf1.global_variables_initializer()

# Passou a sessão para o prompt realizar as funções, e assim aprender 1000 vezes otimizar
with tf1.Session() as ses:
    ses.run(init)
    print('b0='f'{ses.run(b0)}')
    print('b1='f'{ses.run(b1)}')
    for i in range(1000):
        ses.run(treinamento)
    b0_final, b1_final = ses.run([b0,b1])
    print('bo final= ' f'{b0_final}', 'b1 final=' f'{b1_final}')
    
previsoes = b0_final + b1_final*x
print('previsoes escalonadas' f'{previsoes}')

plt.plot(x,y, 'o')
plt.plot(x,previsoes,color ='red')

# Preve um valor do plano para uma pessoa de 40 anos
previsao = scaler_y.inverse_transform(b0_final + b1_final*scaler_x.transform([[40]]))

print('pessoa com 40 anos paga' f'{previsao}')

# Re-escalor as variáveis pois estão em uma métrica diferente
y1 = scaler_y.inverse_transform(y)
previsoes1 = scaler_y.inverse_transform(previsoes)
print ('valores originais' '\n' f'{y1}', '\n' '\n' 'valores finais ' '\n', f'{previsoes1}', )

from sklearn.metrics import mean_absolute_error, mean_squared_error

#utiliza-se para definir quão bom estão as respostas
mae = mean_absolute_error(y1, previsoes1)

# utiliza-se para definir o aprendizado, pois dá melhores resultados
mse = mean_squared_error(y1, previsoes1)
print('\n', mae, mse)
