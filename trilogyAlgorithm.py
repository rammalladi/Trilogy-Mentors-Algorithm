
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.preprocessing import OneHotEncoder as ohe
import numpy.random as random
import numpy.linalg as linalg

def computeGradient(w,x,y):
    pred = np.dot(x, w)
    pred = pred-y
    tot = np.dot(x.transpose(), pred)
    return (tot/len(x[0]))

def computeCost (w,x,y):
    pred = np.dot(x, w)
    tot = (pred-y)**2
    diff = np.sum(tot)
    return diff/len(x[0])

def stochasticGradient(w,x,y,bat):
    start = random.randint(0,len(x)-bat)
    end = start+bat
    x1 = x[start:end,:]
    y1 = y[start:end,:]
    pred = np.dot(x1,w)
    diff = pred-y1
    vec = x1.transpose()
    diff = np.dot(vec,diff)
    return (diff/len(x1))

def standardizeData(x):
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0) 
    return ((x-mean)/std)

def normalizeData(x):
    mean = np.mean(x,axis=0)
    ran = np.max(x,axis=0)-np.min(x,axis=0)
    x = (x-mean)/ran
    return x


raw_data = open("trilogyData.csv")
data = np.loadtxt(raw_data,delimiter=",",skiprows=1, dtype=np.str)

x0 = np.ones((len(data),1))
x = data [:,1:72]
y = data [:,72]
y = y.astype(float)


ohe = ohe(categories = 'auto')
state = ohe.fit_transform(data[:,1].reshape((len(data),1))).toarray().astype(np.float)
grade = ohe.fit_transform(data[:,2].reshape((len(data),1))).toarray().astype(np.float)
x = np.delete(x, 0, axis=1)
x = np.delete(x, 0, axis=1)
arr = np.concatenate((state,grade,x),axis=1)
arr = arr.astype(float)
arr = np.concatenate((x0,arr), axis=1)


x_train, x_test, y_train, y_test = model_selection.train_test_split(arr, y,train_size=0.75,test_size=0.25, random_state=101)

(train_data, train_targets), (test_data, test_targets) = (x_train, y_train), (x_test, y_test)


model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape = (len(train_data[0]),)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


history = model.fit(train_data, train_targets, epochs=100, batch_size=5)  
mae_history = history.history['mae']    
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)


plt.plot(range(0, 100), mae_history, 'r')
plt.title("Mean Absolute Error")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


print("MAE:", test_mae_score)
print("MSE:", test_mse_score)

accuracy = round(100-100*test_mae_score, 2)
print("Accuracy: ",accuracy,"%")

