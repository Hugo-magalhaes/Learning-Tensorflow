from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('mnist/', one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

#import matplotlib.pyplot as plt
#plt.imshow(x_train[0].reshape(28,28), cmap= 'grey' #volta a imagem em escala de cinzas)
x_batch, y_batch = mnist.train.next_batch(64)
neurons_entry = x_train.shape[1]
neurons_hidden1 = int((x_train.shape[1]+y_train.shape[1])/2)
neurons_hidden2 = neurons_hidden1
neurons_hidden3 = neurons_hidden1
neurons_out = y_train.shape[1]

# 784 neuronios na camada de entrada -> 397 na camada oculta 1 -> 397 na 2 -> 397 na 3 -> 10 na saída

w = {'hidden1':tf.Variable(tf.random.normal([neurons_entry, neurons_hidden1])),
     'hidden2': tf.Variable(tf.random.normal([neurons_hidden1, neurons_hidden2])),
     'hidden3':tf.Variable(tf.random.normal([neurons_hidden2, neurons_hidden3])),
     'out':tf.Variable(tf.random.normal([neurons_hidden3, neurons_out]))}

b = {'hidden1':tf.Variable(tf.random.normal([neurons_hidden1])),
    'hidden2': tf.Variable(tf.random.normal([neurons_hidden2])),
    'hidden3': tf.Variable(tf.random.normal([neurons_hidden3])),
        'out':  tf.Variable(tf.random.normal([neurons_out]))}

xph = tf.placeholder('float' , [None, neurons_entry] )
yph = tf.placeholder('float', [None, neurons_out])
def mlp(x, w, bias):
    layer_hidden1= tf.nn.relu(tf.add(tf.matmul(x, w['hidden1']), bias['hidden1']))
    layer_hidden2= tf.nn.relu(tf.add(tf.matmul(layer_hidden1, w['hidden2']), bias['hidden2']))
    layer_hidden3= tf.nn.relu(tf.add(tf.matmul(layer_hidden2, w['hidden3']), bias['hidden3']))
    layer_out= tf.add(tf.matmul(layer_hidden3,w['out']), bias['out'])
    return layer_out

model = mlp(xph,w,b)
erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels = yph))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.0001).minimize(erro)

predictions = tf.nn.softmax(model)
predicts_corrects = tf.equal(tf.argmax(predictions, 1), tf.argmax(yph,1)) # compara os valor true e false
hit_rate = tf.reduce_mean(tf.cast(predicts_corrects, tf.float32))

with tf.Session() as ses:
    ses.run(tf.compat.v1.global_variables_initializer())
    for epochs in range(5000):
        x_batch, y_batch = mnist.train.next_batch(128)#Mudará os pesso somente de 128 em 128 dados
        _,cost = ses.run([optimizer, erro], feed_dict={xph:x_batch,yph:y_batch})
        acc = ses.run([hit_rate],feed_dict={xph:x_batch,yph:y_batch})
        if epochs %100 ==0:
            print('epocas: ' + str((epochs+1)) + 'erro: '+ str(cost) + 'acc: '+ str(acc))
    print( 'treinamneto concluído')
    print(ses.run(hit_rate, feed_dict = {xph:x_test, yph: y_test}))