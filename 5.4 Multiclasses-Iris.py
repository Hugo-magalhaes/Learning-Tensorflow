'''
Base de dados IRIS
SoftMax ajuste 
'''
from sklearn import datasets
iris = datasets.load_iris()
print(iris)

x= iris.data
y = iris.target
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(categories = 'auto')
y = y.reshape(-1,1)
y = onehot.fit_transform(y).toarray()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3)
import tensorflow as tf
import numpy as np
neuron_entry = x.shape[1]
neuron_oculta = int(np.ceil(x.shape[1]+y.shape[1]/2))
neuron_saida= y.shape[1]

w = {'oculta': tf.Variable(tf.random.normal([neuron_entry,neuron_oculta])),
     'saida':tf.Variable(tf.random.normal([neuron_oculta,neuron_saida]))}

b = {'oculta':tf.Variable(tf.random.normal([neuron_oculta])), 
     'saida' :tf.Variable(tf.random.normal([neuron_saida]))}

xph =tf.compat.v1.placeholder('float', [None, neuron_entry])
yph = tf.compat.v1.placeholder('float', [None, neuron_saida])

def mlp(x,w,bias):
    layer_oculta = tf.add(tf.matmul(x,w['oculta']),bias['oculta'])
    layer_oculta_activ = tf.nn.relu(layer_oculta)
    layer_out = tf.add(tf.matmul(layer_oculta_activ, w['saida']), b['saida']) 
    return layer_out

modelo = mlp(xph,w ,b)
erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=modelo, labels= yph))
optim = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.0001).minimize(erro)
batch_size = 8
batch_total = int(len(x_train)/batch_size)

x_batches = np.array_split(x_train, batch_total)

with tf.compat.v1.Session() as ses:
    ses.run(tf.compat.v1.global_variables_initializer())
    for epoca in range(3000):
        erro_medio = 0.0
        batch_total = int(len(x_train)/batch_size)
        x_batches = np.array_split(x_train, batch_total)
        y_batches = np.array_split(y_train, batch_total)
        for i in range(batch_total):
            x_batch, y_batch = x_batches[i], y_batches[i]
            _, custo = ses.run([optim,erro], feed_dict={xph:x_batch, yph:y_batch})
            erro_medio += custo/batch_total
        if epoca%500 == 0:
            print('Epoca:' + str((epoca+1)) + 'erro:' + str(erro_medio))
    w_final, b_final =ses.run([w,b])
    
#Pevisoes
    
predict_test = mlp(xph,w_final,b_final)
with tf.compat.v1.Session() as ses:
    ses.run(tf.compat.v1.global_variables_initializer())
    r1 = ses.run(predict_test, feed_dict ={xph:x_test})
    r2 = ses.run(tf.nn.softmax(r1))
    r3 = ses.run(tf.math.argmax(r2,1))  #retorna a coluna q tem o maior valor 
y_test2 = np.argmax(y_test, 1)
from sklearn.metrics import accuracy_score
taxa_acerto = accuracy_score(y_test2, r3)
print(taxa_acerto)