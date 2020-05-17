import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.preprocessing import OneHotEncoder as ohe

raw_data = open("studentData.csv")
data = np.loadtxt(raw_data,delimiter=",",skiprows=1, dtype=np.str)
x = data [:,1:73]
y = data [:,73]

ohe = ohe(categories = 'auto')
job = ohe.fit_transform(data[:,1].reshape((len(data),1))).toarray().astype(np.float)
"""marital = ohe.fit_transform(data[:,3].reshape((len(data),1))).toarray().astype(np.float)
gender = ohe.fit_transform(data[:,5].reshape((len(data),1))).toarray().astype(np.float)
age = normalize(data[:,0].reshape((len(data),1)).astype(np.float))
col2 = normalize(data[:,2].reshape((len(data),1)).astype(np.float))
col6 = normalize(data[:,6].reshape((len(data),1)).astype(np.float))"""

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,train_size=0.75,test_size=0.25, random_state=101)



(train_data, train_targets), (test_data, test_targets) = (x_train, y_train), (x_test, y_test)
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape = (len(train_data[0]),)))

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(train_data, train_targets,
                        epochs=100, batch_size=1)
mae_history = history.history['mean_absolute_error']       
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
predicted_prices = model.predict(test_data)

mnsqr_error = np.mean(np.power(predicted_prices-test_targets,2))

"""
plot = plt.figure()
ax = plot.add_axes([0.1,0.1,0.8,0.8])
ax.plot(np.arange(len(mae_history)),mae_history,"ro",label="cost")
"""