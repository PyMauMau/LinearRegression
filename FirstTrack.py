import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Suponha que você tenha um DataFrame com os dados
df = pd.DataFrame({
    'commodity_price': [100, 150, 200, 250, 50],
    'fornecedor1_price': [300, 320, 340, 360, 380],
    'fornecedor2_price': [293, 310, 330, 350, 370],
    'fornecedor3_price': [301, 325, 345, 365, 385],
    'fornecedor4_price': [289, 305, 325, 345, 365],
    'fornecedor5_price': [293, 315, 335, 355, 375]
})

# Função para calcular o coeficiente de regressão para um fornecedor
def calcular_coeficiente(df, fornecedor):
    X = df[['commodity_price']].values
    y = df[fornecedor].values
    modelo = LinearRegression().fit(X, y)
    return modelo.coef_[0], modelo.intercept_

# Calcular os coeficientes para cada fornecedor
fornecedores = ['fornecedor1_price', 'fornecedor2_price', 'fornecedor3_price', 'fornecedor4_price', 'fornecedor5_price']
coeficientes = {fornecedor: calcular_coeficiente(df, fornecedor) for fornecedor in fornecedores}

# Definir os cenários de aumento da commodity
precos_commodity = np.linspace(100, 400, 100)

# Plotar os resultados
plt.figure(figsize=(10, 6))

for fornecedor, (coef, intercept) in coeficientes.items():
    precos_produto = intercept + coef * precos_commodity
    plt.plot(precos_commodity, precos_produto, label=fornecedor)

plt.xlabel('Preço da Commodity')
plt.ylabel('Preço do Produto')
plt.title('Cenários de Aumento de Preço da Commodity')
plt.legend()
plt.grid(True)
plt.show()
