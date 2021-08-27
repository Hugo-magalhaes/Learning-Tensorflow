import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
from sklearn.metrics import mean_absolute_error
tf1.disable_v2_behavior()
tf1.disable_eager_execution()

base = pd.read_csv(r'C:\Users\CRose\.spyder-py3\Trabalhos\petr4.csv')
base = base.dropna() #dados que não tenha informações serão retirados
base = base.iloc[:,1].values # o : diz que são todos os registros e 1 da coluna 1 

#plt.plot(base)

'preverá 30 dias a frente dos resgitros'
period = 30
prev_fut = 1 # horizonte

x = base[0:(len(base) - (len(base) % period))] # pega 1230 para treinamento
x_batch = x.reshape(-1,period,1) # None = -1 pq n sabemos quantos valores temos

y = base[1:(len(base) - (len(base) % period)) + prev_fut] 
y_batch = y.reshape(-1,period,1) # empurrou um registro a frente para o atributo previsor

'São usados os valores seguintes para prever o anterior, exp : retira o valor 0 e o preve com valor 1'
x_test = base[-(period+prev_fut):] # para pegar os últimos registros, coloca -
x_test = x_test[:period]
x_test = x_test.reshape(-1,period,1)

y_test = base[-(period):]
y_test = y_test.reshape(-1,period,1)

tf1.reset_default_graph()

entrys = 1
hidden_neurons = 100
out_neurons = 1

xph = tf1.placeholder(tf.float32,[None,period,entrys])
yph = tf1.placeholder(tf.float32, [None,period,out_neurons])
'''
TESTAR DIREITO OS VALORES ENTREGUES POR CADA REDE NEURAL
Observe que há quatro tipos de camada de saida
A primeira com Output e Basic RNN com absolute = 0.167 e squared error = 0.178
A segunda com Output e LSTM com absolute = 0.248 e squared error = 0.148
A terceira com Output e 4 LSTM ( defs) com absolute = 0.279 e squared error = 0.099
E a quarta com Output, 4 LSTM e Dropout com absolute = 4.34 e squared error = 29.8
'''
#cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_neurons, activation = tf.nn.relu)
#cell = tf.contrib.rnn.LSTMCell(num_units = hidden_neurons, activation = tf.nn.relu)
#cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size = 1)

def create_cell():
    return tf.contrib.rnn.LSTMCell(num_units = hidden_neurons, activation = tf.nn.relu)

def create_many_cells():
    return tf1.nn.rnn_cell.MultiRNNCell([create_cell() for i in range(4)])
#    cells = tf.nn.rnn_cell.MultiRNNCell([create_cell() for i in range(4)])
#    return tf.contrib.rnn.DropoutWrapper(cells, output_keep_prob = 0.1)

cell = create_many_cells()
cell =  tf.contrib.rnn.OutputProjectionWrapper(cell, output_size = 1)


out_nn, _ = tf.nn.dynamic_rnn(cell,xph,dtype = tf.float32)
error = tf1.losses.mean_squared_error(labels = xph,predictions =out_nn)
otimizer = tf1.train.AdamOptimizer(learning_rate = 0.001)
trainment = otimizer.minimize(error)

with tf1.Session() as ses:
    ses.run( tf1.global_variables_initializer())
    
    for epoca in range(1000):
        _, cost = ses.run([trainment, error], feed_dict = {xph:x_batch, yph:y_batch})
        if epoca% 100 ==0:
            print(epoca+1,'erro: ', cost)
    prev = ses.run(out_nn, feed_dict ={xph:x_test, yph:y_test})

y_test2 = np.ravel(y_test)
prev2 = np.ravel(prev)

mae = mean_absolute_error(y_test2,prev2)

plt.plot(y_test2, '*', markersize = 10, label = 'Real Value')
plt.plot(prev2, 'o', label = 'Previsions')
plt.legend()
