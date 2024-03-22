import pandas as pd
import numpy as np
from math import pi, sqrt, exp

TRAINING = "train.csv" 
df_train = pd.read_csv(TRAINING, sep=',',header=0)
train = df_train[:10000].groupby(['label']).mean().values
x_test = df_train[10000:].drop("label", axis = 1).values
y_test = df_train[10000:]["label"].values

def g(m, n):
  return ((1/sqrt(2*pi*m*n))*exp(-((m-n)**2)/(2*m*n)))
def gdf(nrows, ncols):
  df = pd.DataFrame(columns=range(1,nrows+1), index=range(1,ncols+1))
  for i in range(nrows):
    for j in range(ncols):
      df.iat[i,j] = g(i+1,j+1)
  return df
metric = gdf(784,784).values

def rho(chi, psi, metric):
  chipsi = pd.DataFrame(chi @ metric @ psi.T)
  chi2 = pd.DataFrame(chi @ metric @ chi.T)
  psi2 = pd.DataFrame(psi @ metric @ psi.T)
  for i in range(chipsi.shape[0]):
    for j in range(chipsi.shape[1]):
      chipsi.iat[i,j] = (chipsi.iat[i,j]/sqrt(chi2.iat[i,i]*psi2.iat[j,j]))
  return chipsi
similarity = rho(train, x_test, metric)
y_pred = similarity.apply(pd.to_numeric).idxmax().values

metric2 = gdf(y_pred.shape[0],y_test.shape[0]).values
score = (y_pred @ metric2 @ y_test.T)/sqrt((y_pred @ metric2 @ y_pred.T)*(y_test @ metric2 @ y_test.T))
print(score)
