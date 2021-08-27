import tensorflow as tf

#Matrices sum
a1= tf.constant([[1,2,3], [4,5,6]], name = 'a1')
print ( type(a1), a1, a1.shape)
b1  = tf.constant([[1,2,3],[4,5,6]], name ='b1')
suma = tf.add(a1, b1)
with tf.Session() as ses:
    print(ses.run(suma))
    print('\n')
    print(ses.run(a1))
    print('\n') # skip one line when print it 
    print(ses.run(b1))


#Matrices sum with different dimension matrices
a2 = tf.constant([[1,2,3],[4,5,6]])
b2 = tf.constant([[1],[2]])
sumb = tf.add(a2, b2)
with tf.Session() as ses:
    print(ses.run(a2), '\n', '\n', ses.run(b2))
    print('\n', ses.run(sumb))