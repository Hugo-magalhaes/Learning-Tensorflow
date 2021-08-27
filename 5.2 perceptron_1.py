import numpy as np
import tensorflow as tf
X = np.array([[0.0,0.0], # valores de entrada
              [0.0,1.0],
              [1.0,0.0],
              [1.0,1.0]])
y = np.array([[0.0],[0.0],[0.0],[1.0]]) # respostas esperadas do perceptron
W = tf.Variable(tf.zeros([2,1], dtype = tf.float64)) # define a varáviel tipo float no tensorflow

init = tf.global_variables_initializer() # inicializador de váriaveis
camada_saida = tf.matmul(X,W) # multplicação de matrizes

def step(x): # função que retorna somente 1 e 0 a partir dos demais valores
    return tf.cast(tf.to_float(tf.math.greater_equal(x,1)), tf.float64)

cms_activ = step(camada_saida) # aplica a função no pesos parametrizando em 0 e 1

# Calculo do erro 1-0 = 1
erro =tf.subtract(y, cms_activ)

delta = tf.matmul(X , erro, transpose_a = True)
# penso(n+1) = peso n{W} + {delta}taxa de aprendizagem*erro*entrada 
treinamento = tf.assign(W,tf.add(W, tf.multiply(delta, 0.1))) # atualização dos pesos


with tf.Session() as ses:
    ses.run(init) # inicializa as variaveis
    ses.run(tf.transpose(X)) # a variável W inicializada
    print(ses.run(camada_saida), '\n')
    ' print(ses.run(tf.to_float(tf.math.greater_equal(1,1))))'# analisa os valores n1 => n2 true, false n1=< n2
    print(ses.run(cms_activ), '\n')
    print(ses.run(erro))
    epoca = 0 # contagem de repetição
    for i in range(15):
        epoca += 1
        erro_total, _=ses.run([erro,treinamento])
        erro_soma = tf.reduce_sum(erro_total) # reduz o erro de todos os parametros 
        print('Época :', epoca, 'Erro:', ses.run(erro_soma))
        if erro_soma.eval() == 0:
            break
    W_final = ses.run(W)
    print(W_final)
    