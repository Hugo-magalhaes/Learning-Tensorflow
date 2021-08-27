import pandas as pd
base = pd.read_csv('census.csv')
def converte(rotulo):
    if rotulo == '>50':
        return 1
    else:
        return 0
base['income']=base['income'].apply(converte)
# Todos os dados menos income é x
x = base.drop('income', axis = 1)
# Somente income é y
y = base['income']
'print(base.age.hist())'

#Para descobrir quais são os valores de uma coluna use:
'x.[''nome da coluna''].unique()'

import tensorflow as tf
age = tf.feature_column.numeric_column('age')
# Criar faixas etárias 
age_cat = [tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80,90])]
nome_colunas_cat = [ 'workclass', 'education',
       'marital-status', 'occupation', 'relationship', 'race', 'native-country']
# Faz o vocabulario de cada coluna
colunas_cat = [tf.feature_column.categorical_column_with_vocabulary_list(key=c,
                                                                         vocabulary_list=
                                                                         x[c].unique())
                                                                         for c in
                                                                         nome_colunas_cat]

nome_colunas_num =   [ 'final-weight', 'education-num',
                      'capital-gain', 'capital-loos', 'hour-per-week'] 
colunas_num =[tf.feature_column.numeric_column(key =c) for c in nome_colunas_num]
colunas = age_cat + colunas_cat + colunas_num


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.3)
funcao_train = tf.estimator.inputs.pandas_input_fn(x = x_train,y = y_train, batch_size = 32
                                                   , num_epochs = None, shuffle = True)
classific = tf.estimator.LinearClassifier(feature_columns= colunas)
classific.train(input_fn =funcao_train, steps =10000)


funcao_prev = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size = 32, shuffle =False)
prev = classific.predict(input_fn=funcao_prev)
prev_final = []
print(list(prev))
for p in classific.predict(input_fn = funcao_prev):
    prev_final.append(p['class_ids'])
from sklearn.metrics import accuracy_score
taxa_acerto = accuracy_score(y_test, prev_final)
print(taxa_acerto)

