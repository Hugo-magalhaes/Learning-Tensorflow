import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

mnist = input_data.read_data_sets('mnist/', one_hot = False) #

# Aqui são definidos os parametros para entrada e resposta no treinamento e teste
x_treina =mnist.train.images
y_treina = mnist.train.labels
y_treina = np.asarray(y_treina, dtype = np.int32)
x_test = mnist.test.images
y_test = mnist.test.labels
y_test = np.asarray(y_test,dtype =np.int32)

# Cria-se a função que será a CNN com as convoluções e poolings para reduzir a dimensão
'O parametro mode é o usuario que define = Treinamento, teste ou previsão'
def cria_rede(features,labels, mode):
#       batch_size, largura, altura, canais de cores
    entrada = tf.reshape(features['X'], [-1,28,28,1])
#    recebe [batch_size, 28,28,1]
#    retorna [batch_size, 28,28,32]
    convolucao1= tf.layers.conv2d(inputs = entrada, filters = 32, kernel_size = [5,5], 
                                  activation = tf.nn.relu, padding = 'same')
#    recebe [batch_size, 28,28,32]
#    retorna [batch_size, 14,14,32]
    pooling1 = tf.layers.max_pooling2d(inputs = convolucao1, pool_size = [2,2], strides =2)
    
#    recebe [batch_size, 14,14,32]
#    retorna [batch_size, 14,14,64]
    convolucao2 =tf.layers.conv2d(inputs = pooling1, filters = 64, kernel_size= [5,5],
                                  activation = tf.nn.relu, padding = 'same')
#    recebe [batch_size, 14,14,32]
#    retorna [batch_size, 7,7,64]
    pooling2 = tf.layers.max_pooling2d(inputs = convolucao2, pool_size = [2,2], strides =2)
#    recebe [batch_size, 7,7,64]
#    retorna [batch_size, 3136]    
    flattening = tf.reshape(pooling2, [-1,7*7*64])
    
#   3136 entradas -> 1024 oculta -> 10 saída
#    recebe [batch_size, 3136]
#    retorna [batch_size, 1024]
    densa = tf.layers.dense(inputs = flattening, units = 1024, activation = tf.nn.relu)

#    pode usar a funçao dropout para diminuir o overfitting
    dropout = tf.layers.dropout(inputs = densa, rate = 0.2,
                                training = mode == tf.estimator.ModeKeys.TRAIN) 
# necessário testar a qtd a ser excluida pelo dropout para a eficiência do processo, sem generalizar
    
#    recebe [batch_size, 1024]
#    retorna [batch_size, 10]
    saida = tf.layers.dense(inputs = dropout, units = 10 ) 
    previsoes = tf.argmax(saida, axis = 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode= mode, predictions = previsoes)

    erro = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = saida)
# Os pesos e os erros só são calculados na etapa de treinamento por isso um if
    if mode  == tf.estimator.ModeKeys.TRAIN:
        otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)
        treinamento = otimizador.minimize(erro, global_step = tf.train.get_global_step())
# global_step pega o step anterior para gerar o próximo step
# A saida é defiida pela interpretação do da funca_treina que precisa de um estimatorspec
        return tf.estimator.EstimatorSpec(mode = mode, loss = erro, train_op = treinamento)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics_ops = {'accuracy': tf.metrics.accuracy(labels = labels,
                                                            predictions = previsoes)}
        return tf.estimator.EstimatorSpec(mode = mode, loss = erro, 
                                          eval_metric_ops = eval_metrics_ops)
' Etapa de treinamento da CNN criada'
# Define os parametros da CNN, como foi feito a função cria_rede, o estimator só precisa dela
classificador = tf.estimator.Estimator(model_fn = cria_rede)
# Num_epochs esta em None porque não é ainda o teste que precisa de repetição
# shuffle determina se pega valores aleatórios = True, False = não aleatórios
func_treina = tf.estimator.inputs.numpy_input_fn(x = {'X' : x_treina}, y = y_treina, 
                                                 batch_size = 128, num_epochs =None, shuffle = True)

'Ao usar o tf.train, definimos o parametro mode para treinamento da CNN'
classificador.train(input_fn =func_treina, steps = 200)

'Etapa de teste da CNN'

func_test = tf.estimator.inputs.numpy_input_fn(x = {'X' : x_test}, y = y_test, num_epochs = 1
                                                    ,shuffle = True)
resultados = classificador.evaluate(input_fn = func_test)

'Etapa de previsão'

x_image_test = x_test[0]
x_image_test = x_image_test.reshape(1,-1)
func_previsao = tf.estimator.inputs.numpy_input_fn(x = {'X': x_image_test}, shuffle = False)
pred = list(classificador.predict(input_fn = func_previsao))

' Aqui você observar se o resultado está certo, e variando x_test[], observa outros resultados'
plt.imshow(x_treina[0].reshape(28,28), cmap = 'gray') # observa a primeira imagem dos dados de teste
plt.title('Valor real: ' + str(y_treina[0]) + ' previsão:' + str(pred[0]))
# a resposta que a rede neural deve ter
