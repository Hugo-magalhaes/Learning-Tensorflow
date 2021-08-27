'''
Função sigmoide --> usada para definir regiões não lineares ( função xor) 
y = 1/(1+e^(-x))
Função xor definida como:
    Dado1 Dado2 Classe(resposta)
    0     0     0
    0     1     1
    1     0     1
    1     1     0
Observe que a função xor não responde a lógica linear 
e por isso é necessário a função sigmoide para encontrar os pesos certos para a rede neural
Derivada ou descida do gradiente =
d = y(1-y)
fazendo a derivada da função sigmoide com o dado encontrado

'''
# as funções de ativação para definir funções que não lineares para redes neurais
import numpy as np

# Para problemas linearmente separáveis
def StepF(soma):
    if soma >= 1:
        return 1
    return 0
teste = StepF(-1)

# problemas de classificação binária
def sigmoidF(soma):
    return 1/(1+np.exp(-soma))
teste1 = sigmoidF(0.358)

# Problemas com valores entre -1 e 1 e classificação
def TanHF(soma):
    return (np.exp(soma)-np.exp(-soma))/(np.exp(soma)+np.exp(-soma))
teste2 = TanHF(0.358)

# redes neurais convulcionais e muitas camadas 
def Relu(soma):
    if soma >= 0:
        return soma
    return 0
teste3 = Relu(0.358)

# Usada para regressão
def linearF(soma):
    return soma
teste4 = linearF(30)

# Problemas de classificação com mais de duas classes
def softmaxF(x):
    ex = np.exp(x)
    return ex / ex.sum()
valores = [5.0, 2.0, 1.3]
print(softmaxF(valores))
