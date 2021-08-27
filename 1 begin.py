import tensorflow.compat.v1 as tf

#constant sum
v1 = tf.constant(1)
v2 = tf.constant(2)
sum1 = v1 + v2 
print(sum1)
with tf.Session() as sess:
    s = sess.run(sum1)
print(s)


#strings sum

text1=tf.constant('text 1')
text2 = tf.constant('text 2')
print(type(text1))
with tf.Session() as sess:
    s = sess.run(text1 + text2)
print(s)



#Naming a cell after create it makes it... 
v1 = tf.constant(15, name = 'v1')
print(v1)
soma = tf.Variable( v1 + 5, name = 'v1')
print(soma, type(soma))

#Always is necessary initialize the variables for action happens
init = tf.global_variables_initializer()
with tf.Session() as ses:
    ses.run(init)
    print(ses.run(sum1))
