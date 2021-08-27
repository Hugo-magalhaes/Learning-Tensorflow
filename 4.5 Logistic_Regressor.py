import pandas as pd
base = pd.read_csv('census.csv')
print(base['income'].unique())
x =base.iloc[:,0:14].values
y = base.iloc[:,14].values

# Transformação os atributos que não são numéricos
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x[:,1] = label_encoder.fit_transform(x[:,1])
x[:,3] = label_encoder.fit_transform(x[:,3])
x[:,5] = label_encoder.fit_transform(x[:,5])
x[:,6] = label_encoder.fit_transform(x[:,6])
x[:,7] = label_encoder.fit_transform(x[:,7])
x[:,8] = label_encoder.fit_transform(x[:,8])
x[:,9] = label_encoder.fit_transform(x[:,9])
x[:,13] = label_encoder.fit_transform(x[:,13])

# Escaolnamento de todas as variavéis pois estão em medidas diferentes
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
z = scaler_x.fit_transform(x)

# Criando as variaveis de teste e treinamento
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(z, y, test_size =0.3)

#Regressão e o valor preditado
from sklearn.linear_model import LogisticRegression
classific = LogisticRegression(max_iter=10000)
classific.fit(x_train, y_train)
previsoes = classific.predict(x_test)


# Observa quantos acertos a predição faz
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,previsoes)
print(accuracy)