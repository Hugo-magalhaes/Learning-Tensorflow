import tensorflow as tf
tf.compat.v1.reset_default_graph()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist/', one_hot = True)  

Red   = "\033[1;31m"  
Green  = "\033[1;32m"
Cyan  = "\033[1;36m"


import numpy as np
image1 = np.arange(0,784).reshape(28,28)
image2 = np.random.normal(size =784).reshape(28,28)
noise_ph = tf.compat.v1.placeholder(tf.float32, [None,100])

def generator(noise,reuse = None):
    with tf.compat.v1.variable_scope('generator', reuse = reuse):
#        110 -> 128 -> 128 -> 784
        hidden_layer1 = tf.nn.relu(tf.layers.dense(inputs = noise, units = 128))
        hidden_layer2 = tf.nn.relu(tf.layers.dense(inputs = hidden_layer1, units =128))
        out_layer = tf.layers.dense(inputs = hidden_layer2, units = 784, activation = tf.nn.tanh)
        return out_layer

real_images_ph = tf.compat.v1.placeholder(tf.float32, [None, 784])

def discriminador(x, reuse= None):
    with tf.compat.v1.variable_scope('discriminator', reuse= reuse):
        # 784 -> 128->128 -> 1
        hidden_layer1 = tf.nn.relu(tf.layers.dense(inputs = x, units = 128))
        hidden_layer2 = tf.nn.relu(tf.layers.dense(inputs = hidden_layer1, units = 128))
        logits = tf.layers.dense(inputs = hidden_layer2, units = 1) # logit é resposta não normalizada
        return logits

real_images_logits = discriminador(real_images_ph)
'Sempre que der algum problema, reinicie o True abaixo com False'
noise_images_logits = discriminador(generator(noise_ph), reuse= True) 

error_discriminador_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = real_images_logits, 
                                                                                  labels = tf.ones_like(
                                                                                          real_images_logits)*(0.9))) 
error_discriminador_noise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = noise_images_logits,
                                                                                   labels = tf.zeros_like(
                                                                                           noise_images_logits)))
erro_discriminador = error_discriminador_real + error_discriminador_noise
erro_generator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = noise_images_logits,
                                                                        labels = tf.ones_like(noise_images_logits)))
variables = tf.compat.v1.trainable_variables()

variables_discriminador =[v for v in variables if 'discriminator' in v.name]
variables_generator =[v for v in variables if 'generator' in v.name]
print(variables_generator)
#Se não o var list ele mistura todas as variaveis
training_discriminador = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.001).minimize(erro_discriminador,
                                                   var_list = variables_discriminador)
training_generator = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.001).minimize(erro_generator,
                                               var_list = variables_generator)

batch_size = 100
amostra_test = []


with tf.compat.v1.Session() as ses:
    ses.run(tf.compat.v1.global_variables_initializer())
#    noise_test = np.random.uniform(-1,1,size=(1,100))
#    amostra = ses.run(generator(noise_ph,reuse= True), feed_dict={noise_ph:noise_test})

#    batch = mnist.train.next_batch(100)
#    batch_images = batch[0].reshape((100,784))
#    batch_images *=2 - 1
#    r = ses.run(discriminador(real_images_ph, True), feed_dict = {real_images_ph:batch_images})
#    r2 = ses.run(tf.nn.sigmoid(r))

#    ex = tf.constant([[1,2],[3,4]])
#    print(ses.run(tf.ones_like(ex)))
    for epoca in range(50):
        number_batches = mnist.train.num_examples//batch_size
        for i in range(number_batches):
            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0].reshape((100,784))
            batch_images *=2 - 1
            batch_noise = np.random.uniform(-1,1, size = ( batch_size ,100))
            _, costd = ses.run([training_discriminador, erro_discriminador], feed_dict = 
                               {real_images_ph: batch_images, noise_ph : batch_noise})
            _,costg = ses.run([training_generator, erro_generator], feed_dict = {noise_ph: batch_noise})
        print(f' {Red} época  = { epoca + 1 }, {Green} erro D  = {costd}, {Cyan} erro G = {costg}')
        noise_test = np.random.uniform(-1,1, size = (1,100)) 
        generate_image = ses.run(generator(noise_ph, reuse = True), feed_dict = {noise_ph:noise_test})
        amostra_test.append(generate_image) # append adiciona cada imagem do for gerada na lista amostra
    
#import matplotlib.pyplot as plt
#plt.imshow(amostra_test[49].reshape(28,28), cmap = 'Greys')
        
        