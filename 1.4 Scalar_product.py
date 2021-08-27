from numpy.core.fromnumeric import product
import tensorflow as t
a = t.constant([[-1.,7.,5.]], name = 'entradas')
b = t.constant([[0.8,0.1,0]], name = 'pesos')
Product = t.multiply(a,b)
suma = t.reduce_sum(Product)
with t.Session() as s:
    print(s.run(a))
    print('\n')
    print(s.run(b))
    print('\n')
    print(s.run(Product))
    print('\n')
    print(s.run(suma))
    