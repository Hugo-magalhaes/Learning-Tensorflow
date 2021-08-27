import tensorflow as tf

# Vector sum with a scalar 
vector = tf.constant([5, 10 , 15], name = 'vertor')
print(type(vector), vector)

soma = tf.Variable(vector + 5, name='soma')
ini =tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as ses:
    ses.run(ini)
    print(ses.run(soma))

# Set up a scalar to repeat sum 5x
v = tf.Variable(0, name = 'valor')
ini2=tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as ses:
    ses.run(ini2)
    for i in range(5):
        v= v+1 #how the value beahves a int not is possible do v = +1
        print(ses.run(v))


# Vectors sum
a = tf.constant([9,8,7], name='a')
b = tf.constant([1,2,3], name = 'b')
adic = a+b

with tf.compat.v1.Session() as ses:
    print(ses.run(adic))

    