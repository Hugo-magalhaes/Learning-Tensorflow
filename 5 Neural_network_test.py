import numpy as np
classe = np.array([1,0,1,0])
valor_calc = np.array([0.3, 0.02,0.89,0.32])
import sklearn.metrics as skl
mae = skl.mean_absolute_error(classe, valor_calc)
mse = skl.mean_squared_error(classe, valor_calc)
rmse = np.sqrt(skl.mean_squared_error(classe,valor_calc))
print(mae, mse, rmse)