import tensorflow as tf

#Matrix multiplication
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[-1,3],[4,2]])
multi = tf.matmul(a,b)
with tf.Session() as ses:
    print(ses.run(a),'\n','\n', ses.run(b))
    print('\n')
    print(ses.run(multi))
    
#Inverse multiplication matrix
multi2 = tf.matmul(b,a)
with tf.Session() as ses:
    print(ses.run(multi2))

#Showing how matrices multiplication happens with the dimension of opposite matrices ([2,3]x[3,2])
a1 = tf.constant([[2,3],[0,1],[-1,4]])
b1  = tf.constant([[1,2,3],[-2,0,4]])
multi3 = tf.matmul(a1,b1)
with tf.Session() as ses:
    print(ses.run(multi3))
    print('\n')
    print(ses.run(a1))
    print('\n')
    print(ses.run(b1))
