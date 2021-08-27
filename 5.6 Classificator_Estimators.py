import pandas as pd
import tensorflow as tf

base = pd.read_csv('census.csv')
def converte_classe(rotulo):
    if rotulo == ' >50K':
        return 1
    else:
        return 0

base.income = base.income.apply(converte_classe)
x = base.drop('income', axis = 1)
y = base.income

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)

#Categorizando as colunas com valores em string
workclass = tf.feature_column.categorical_column_with_hash_bucket(key = 'workclass', hash_bucket_size = 100)
education = tf.feature_column.categorical_column_with_hash_bucket(key = 'education', hash_bucket_size = 100)
occupation = tf.feature_column.categorical_column_with_hash_bucket(key = 'occupation', hash_bucket_size = 100)
relationship = tf.feature_column.categorical_column_with_hash_bucket(key = 'relationship', hash_bucket_size = 100)
race = tf.feature_column.categorical_column_with_hash_bucket(key = 'race', hash_bucket_size = 100)
country = tf.feature_column.categorical_column_with_hash_bucket(key = 'native-country', hash_bucket_size = 100)
marital_status = tf.feature_column.categorical_column_with_hash_bucket(key = 'marital-status', hash_bucket_size = 100)
sex = tf.feature_column.categorical_column_with_vocabulary_list(key = 'sex', vocabulary_list = [' Male', ' Female' ])

# Formatando elas para um modelo de colunas que o tensor veja em 3 dimensoes 

emb_workclass = tf.feature_column.embedding_column(workclass, dimension = 9)
emb_education = tf.feature_column.embedding_column(education, dimension = len(base.education.unique()))
emb_occupation = tf.feature_column.embedding_column(occupation, dimension = len(base.occupation.unique()))
emb_race = tf.feature_column.embedding_column(race, dimension = len(base.race.unique()))
emb_country = tf.feature_column.embedding_column(country, dimension = len(base['native-country'].unique()))
emb_marital = tf.feature_column.embedding_column(marital_status,dimension = len(base['marital-status'].unique()))
emb_sex = tf.feature_column.embedding_column(sex, dimension = len(base.sex.unique()))
emb_relationship = tf.feature_column.embedding_column(relationship, dimension = len(base.relationship.unique()))

#Usando base.(nome).sid/mean achou o desvio padrão e média de cada função
def padroniza_age(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(38.58)), tf.constant(13.64))

def padroniza_finalweight(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(189778.36)), tf.constant(105549.977))

def padroniza_education(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(10.08)), tf.constant(2.57))

def padroniza_capitalgain(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(1077.64)), tf.constant(7385.29))

def padroniza_capitalloos(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(87.30)), tf.constant(402.96))

def padroniza_hour(valor):
    return tf.divide(tf.subtract(tf.cast(valor, tf.float32), tf.constant(40.43)), tf.constant(12.34))

# Categorizando colunas com valores numéricos
age = tf.feature_column.numeric_column(key = 'age', normalizer_fn = padroniza_age)
final_weight = tf.feature_column.numeric_column(key = 'final-weight', normalizer_fn = padroniza_finalweight)
education_num = tf.feature_column.numeric_column(key = 'education-num', normalizer_fn = padroniza_education)
capital_gain = tf.feature_column.numeric_column(key = 'capital-gain', normalizer_fn = padroniza_capitalgain)
capital_loos = tf.feature_column.numeric_column(key = 'capital-loos', normalizer_fn = padroniza_capitalloos)
hour = tf.feature_column.numeric_column(key = 'hour-per-week', normalizer_fn = padroniza_hour)

# Criando a tabela de dados
'colunas = [age, workclass, education, occupation, relationship, race,country,marital_status,sex,final_weight,education_num, capital_gain,capital_loos,hour]'

colunas_rna = [age, emb_workclass, emb_education, emb_occupation, emb_relationship, emb_race,
               emb_country,emb_marital,emb_sex,
           final_weight,education_num, capital_gain,capital_loos,hour]

#Função que treinará a rede neural
func_train = tf.estimator.inputs.pandas_input_fn(x = x_train,y = y_train, batch_size= 32
                                                 , num_epochs = None, shuffle = True)
#Classificadores com a coluna não embbedada
'classificador = tf.estimator.DNNClassifier(hidden_units = [8,8], feature_columns= colunas, n_classes=2 )'
'classificador.train(input_fn = func_train)'
 
classificador = tf.estimator.DNNClassifier(hidden_units = [8,8], feature_columns= colunas_rna, n_classes=2 )
classificador.train(input_fn = func_train, steps = 10000)

func_test = tf.estimator.inputs.pandas_input_fn(x=x_test, y = y_test, batch_size=32,
                                                num_epochs = 1, shuffle = True)
classificador.evaluate(input_fn = func_test)