from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('minist/', one_hot = True)
x = mnist.train.images

# 784 -> 128 -> 64 -> 128 -> 784
#encode
neurons_entry = 784
neurons_hidden1 = 128
#image decodify
neurons_hidden2 = 64
#decode
neurons_hidden3 = neurons_hidden1
neurons_out = neurons_entry

import tensorflow as tf
tf.reset_default_graph()
xph = tf.placeholder(tf.float32, [None, neurons_entry])

initializer = tf.variance_scaling_initializer()

w = {'encoder_hidden1': tf.Variable(initializer([neurons_entry, neurons_hidden1])),
     'encoder_hidden2': tf.Variable(initializer([neurons_hidden1, neurons_hidden2])),
     'decoder_hidden3': tf.Variable(initializer([neurons_hidden2, neurons_hidden3])),
     'decoder_out': tf.Variable(initializer([neurons_hidden3, neurons_out]))
     }

b = {'encoder_hidden1': tf.Variable(initializer([neurons_hidden1])),
     'encoder_hidden2': tf.Variable(initializer([neurons_hidden2])),
     'decoder_hidden3': tf.Variable(initializer([neurons_hidden3])),
     'decoder_out': tf.Variable(initializer([neurons_out]))
     }

layer_hidden1 = tf.nn.relu(tf.add(tf.matmul(xph,w['encoder_hidden1']), b['encoder_hidden1']))
layer_hidden2 = tf.nn.relu(tf.add(tf.matmul(layer_hidden1, w['encoder_hidden2']), b['encoder_hidden2']))
layer_hidden3 = tf.nn.relu(tf.add(tf.matmul(layer_hidden2, w['decoder_hidden3']), b['decoder_hidden3']))
layer_out = tf.nn.relu(tf.add(tf.matmul(layer_hidden3, w['decoder_out']), b['decoder_out']))
error = tf.losses.mean_squared_error(xph,layer_out)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate =.001)
training = optimizer.minimize(error)
batch_size = 128

with tf.compat.v1.Session() as ses:
    ses.run(tf.compat.v1.global_variables_initializer())
    for epocas in range(50):
        num_epocas = mnist.train.num_examples//batch_size
        for i in range(num_epocas):
            x_batch,_ = mnist.train.next_batch(batch_size)
            cost,_ = ses.run([error,training], feed_dict = {xph:x_batch})
        print('epoca:' + str(epocas+1) + 'erro:' + str(cost))
    image_codify = ses.run(layer_hidden2, feed_dict = {xph:x})
    image_decodify = ses.run(layer_out, feed_dict = {xph:x})
    
import numpy as np
num_images = 5
image_test = np.random.randint(x.shape[0], size = num_images)

import matplotlib.pyplot as plt
plt.figure(figsize = (8,8))
for i, indice_image in enumerate(image_test):
    eixo = plt.subplot(10,5, i+1)
    plt.imshow(x[indice_image].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    
    eixo = plt.subplot(10,5, i+1+num_images)
    plt.imshow(image_codify[indice_image].reshape(8,8))
    plt.xticks(())
    plt.yticks(())
    
    eixo = plt.subplot(10,5, i+1+num_images*2) 
    plt.imshow(image_decodify[indice_image].reshape(28,28))
    plt.xticks(())
    plt.yticks(())