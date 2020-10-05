#! /usr/bin/env python
# @carolio7

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn import metrics


data = pd.read_csv('hubble_data.csv')

"""data.plot(x='distance',y='recession_velocity', style='o')
plt.title('courbe')
plt.xlabel('Distance')
plt.ylabel('Vitesse')
plt.show()"""

"""plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(data['recession_velocity'])
plt.show()"""

x = data['distance'].values.reshape(-1,1)
y = data['recession_velocity'].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

"""print('La variable x_train est de type : ',type(x_train))
print('Les valeur de x_train : ', x_train)
print('Les valeurs de x_test :', x_test)

print('Les valeurs de y_train : ', y_train)
print('Les valeur de y_test : ', y_test)"""

regresseur = LinearRegression()
regresseur.fit(x_train, y_train)

print(regresseur.intercept_)
print(regresseur.coef_)

y_pred = regresseur.predict(x_test)

df = pd.DataFrame({'Actuel': y_test.flatten(), 'Predit': y_pred.flatten()})


df.plot(kind='bar', figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
