import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
x = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([[1],[0],[0],[1]])
neuronios_entrada = 2
neuronios_oculta = 3 
neuronios_saida = 1 
w = {'oculta':tf.Variable(tf.random.normal([neuronios_entrada,neuronios_oculta, ]), name= "w_oculta"),
    'saida': tf.Variable(tf.random.normal([neuronios_oculta, neuronios_saida]), name = "w_saida")}

b = {'oculta': tf.Variable(tf.random.normal([neuronios_oculta]), name = "b_oculta"),
     'saida': tf.Variable(tf.random.normal([neuronios_saida]), name = "b_saida")}

xph = tf.placeholder(tf.float32, [4,neuronios_entrada],  name = 'xph')
yph = tf.placeholder(tf.float32, [4,neuronios_saida], name = 'yph')

camada_oculta = tf.add(tf.matmul(xph, w['oculta']), b['oculta'])
camada_oculta_ativacao = tf.sigmoid(camada_oculta)
camada_saida = tf.add(tf.matmul(camada_oculta_ativacao, w['saida']), b['saida'])
camada_saida_ativacao =  tf.sigmoid(camada_saida)
erro = tf.losses.mean_squared_error(yph, camada_saida_ativacao)
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.3).minimize(erro)

# O inicializador de variavel deve vir sempre depois que declarado a variavel
init = tf.global_variables_initializer()

distribuicao = np.random.normal(size=500)
#import seaborn as sea
#print(sea.distplot(distribuicao))


with tf.Session() as ses:
    ses.run(init)
#    print(ses.run(w['oculta']))
#    print(ses.run(w['saida']))
#    print(ses.run(b['oculta']))
#    print(ses.run(b['saida']))
#    print(ses.run(camada_oculta, feed_dict ={xph: x}))
#    print(ses.run(camada_oculta_ativacao, feed_dict ={xph: x}))
#    print(ses.run(camada_saida, feed_dict ={xph: x}))
#    print(ses.run(camada_saida_ativacao, feed_dict ={xph: x}))
    for epocas in range(10000):
        erro_medio = 0
        _, custo = ses.run([otimizador, erro], feed_dict = {xph:x, yph:y})
        if epocas % 200 == 0:
#            print(custo)
            erro_medio += custo/4
    print(erro_medio)
    w_final, b_final = ses.run([w,b])
print(w_final ,'\n', b_final)
'teste'



camada_oculta_test = tf.add(tf.matmul(xph, w_final['oculta']), b_final['oculta'])
camada_oculta_ativacao_test = tf.sigmoid(camada_oculta_test)
camada_saida_test = tf.add(tf.matmul(camada_oculta_ativacao_test, w_final['saida']), b_final['saida'])
camada_saida_ativacao_test = tf.sigmoid(camada_saida_test)

with tf.Session() as ses:
    ses.run(init)
    print(ses.run(camada_saida_ativacao_test, feed_dict ={xph: x}))






