import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame(
    {'X1': pd.Series([20, 20, 20, 40, 40, 40, 60, 60, 60]),
     'X2': pd.Series([80, 100, 120, 80, 100, 120, 80, 100, 120]),
     'Y': pd.Series([100, 150, 200, 200, 250, 300, 300, 350, 400])}
)

reg = LinearRegression()
reg.fit(df.ix[:, ['X1', 'X2']], df['Y'])

print reg.coef_, reg.rank_, reg.intercept_
print reg.predict([40, 120])
