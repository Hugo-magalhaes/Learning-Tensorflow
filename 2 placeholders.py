import tensorflow.compat.v1 as t
'If you trying to download packages for python use pip install instead of conda install'
# showing hot to use sum functions of placeholders
p = t.placeholder('float', None)
operac = p+2
with t.Session() as s:
    result = s.run(operac, feed_dict ={p:[1,2,3]})
    print(result)
    
# how to use multiplication functions and gives size 5 for placeholder 
p2= t.placeholder('float', [None, 5])
operac2 = p2 * 5
with t.Session() as s:
    dados =[[1,2,3,4,5],[6,7,8,9,10]]
    result = s.run(operac2, feed_dict={p2:dados})
    print(result)