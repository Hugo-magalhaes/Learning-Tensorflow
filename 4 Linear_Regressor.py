import numpy as np 
x = np.array([[18],[23],[28],[33],[38],[43],[48],[53], [58],[63]])
y = np.array([[871],[1132],[1042],[1356],[1488],[1638],[1569],[1754],[1866],[1900]])

import matplotlib.pyplot as plt

#procura os valores para a regressão linear
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

#b0
print('b0=' f'{regressor.intercept_}')
#b1
print('b1=' f'{regressor.coef_}')

previsao1 = regressor.intercept_ + regressor.coef_*40
print(previsao1)

previsao2 = regressor.predict([[40]])
print('\n', previsao2)


previsoes = regressor.predict(x)
print('\n', previsoes)
print('\n')

#formato de calcular erro básico diretamente
resultado = abs(y-previsoes).mean()
print('erro absoluto direto =' f'{resultado}')
print('\n')

# Calcula o erro básico por módulo e também o erro quadratico
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y, previsoes)
mse = mean_squared_error(y,previsoes)
print('erro absoluto= ' f'{mae}' ',erro= ' f'{mse}')


plt.plot(x,y,'o')
plt.plot(x,previsoes, color = 'red')
plt.title('Regressão Linear')
plt.xlabel('idade')
plt.ylabel('Custo')
