import numpy as np
import tensorflow as tf
import pandas as pd

base = pd.read_csv('credit_data.csv')
base = base.drop('i#clientid', axis = 1)
base = base.dropna()

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
base[['income', 'age', 'loan',]] = scaler_x.fit_transform(base[['income', 'age', 'loan']])

x= base.drop('c#default', axis = 1)
y = base['c#default']

colunas = [tf.feature_column.numeric_column(key = column) for column in x.columns]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .3, random_state = 0)

func_train = tf.compat.v1.estimator.inputs.pandas_input_fn(x = x_train,
                                                 y = y_train,
                                                 batch_size = 8,
                                                 num_epochs = None,
                                                 shuffle = True) 
classifier = tf.compat.v1.estimator.DNNClassifier(feature_columns = colunas, hidden_units = [4,4])
classifier.train(input_fn= func_train, steps = 1000)

func_test = tf.compat.v1.estimator.inputs.pandas_input_fn(x = x_test,
                                                           y = y_test,
                                                           batch_size = 8,
                                                           num_epochs = 1000,
                                                           shuffle = False)
test_metrics = classifier.evaluate(input_fn = func_test, steps = 1000)

neurons_entry = 3
neurons_hidden = 2
neurons_out = neurons_entry
xph = tf.compat.v1.placeholder(tf.float32, shape = [None, neurons_entry])

from tensorflow.contrib.layers import fully_connected
layer_hidden = fully_connected(inputs = xph, num_outputs = neurons_hidden, activation_fn = None )
layer_out = fully_connected(inputs = layer_hidden, num_outputs = neurons_out)
error = tf.losses.mean_squared_error(labels = xph, predictions = layer_out)
optimizer = tf.compat.v1.train.AdamOptimizer(.01)
train= optimizer.minimize(error)

with tf.compat.v1.Session() as ses:
  ses.run(tf.compat.v1.global_variables_initializer())
  for epoca in range(1000):
    cost,_ = ses.run([error,train], feed_dict ={xph:x})
    if epoca % 100 == 0:
        print('erro: ' + str(cost))
  x2d_encode = ses.run(layer_hidden, feed_dict = {xph:x})
  x3d_decode = ses.run(layer_out, feed_dict = {xph:x})
  
x2 = scaler_x.inverse_transform(x)
x3d_decode2 = scaler_x.inverse_transform(x3d_decode)

from sklearn.metrics import mean_absolute_error as mae
mae_income = mae(x2[:,0],x3d_decode2[:,0])
mae_age = mae(x2[:,1], x3d_decode2[:,1])
mae_loan = mae(x2[:,2], x3d_decode2[:,2])

x_encode = pd.DataFrame({'atributo1' : x2d_encode[:,0], 'atributo2' : x2d_encode[:,1],'classe' : y})

colunas = [tf.feature_column.numeric_column(key = column) for column in x_encode.columns]
x_train,x_test,y_train,y_test = train_test_split(x_encode,y,test_size = .3, random_state = 0)

func_train = tf.compat.v1.estimator.inputs.pandas_input_fn(x = x_train,
                                                 y = y_train,
                                                 batch_size = 8,
                                                 num_epochs = None,
                                                 shuffle = True) 
classifier = tf.compat.v1.estimator.DNNClassifier(feature_columns = colunas, hidden_units = [4,4])
classifier.train(input_fn= func_train, steps = 1000)

func_test = tf.compat.v1.estimator.inputs.pandas_input_fn(x = x_test,
                                                           y = y_test,
                                                           batch_size = 8,
                                                           num_epochs = 1000,
                                                           shuffle = False)
test_metrics = classifier.evaluate(input_fn = func_test, steps = 1000)