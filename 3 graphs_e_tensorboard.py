import tensorflow as tf


tf.reset_default_graph()

#Cria grafos de forma mais análitca, mostrando todas as operações
a = tf.add(2,2, name='add')
b = tf.multiply(a,3, name ='mult1')
c = tf.multiply(b, a, name='mult2')  

#Organiza grafos de forma mais abreaviada
with tf.name_scope('Operacoes'):
    with tf.name_scope('Escopa_A'):
        d = tf.add(2,2,name='add')
    with tf.name_scope('Escopo_B'):
            e = tf.multiply(a,3,name = 'mult1')
            f = tf.multiply(b,a, name = 'mult2')
#Esta sessão gera o grafo tanto de a,b,c quanto de d,e,f e momento diferentes
with tf.Session() as ses:
    wri = tf.summary.FileWriter('.', ses.graph)
    print(ses.run(c))
    wri.close()

#Mostra a sessão onde o grafo padrão fica
tf.get_default_graph()
#Define uma nova região de grafo
graf = tf.Graph()
#Define graf como grafo padrão
with graf.as_default():
    print(graf is tf.get_default_graph())
