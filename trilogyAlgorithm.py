
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.preprocessing import OneHotEncoder as ohe
import numpy.linalg as linalg
import numpy.random as random


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
    minimum = np.min(x,axis=0)
    maximum = np.max(x,axis=0)
    rang = maximum-minimum
    z = (x-minimum)/rang
    return z


raw_data = open("trilogyData.csv")
data = np.loadtxt(raw_data,delimiter=",",skiprows=1, dtype=np.str)

x0 = np.ones((len(data),1))
x = data [:,1:72]
y = data [:,72]
y = y.astype(float)


ohe = ohe(categories = 'auto')
state = ohe.fit_transform(data[:,1].reshape((len(data),1))).toarray().astype(np.float)
grade = ohe.fit_transform(data[:,2].reshape((len(data),1))).toarray().astype(np.float)

cols = data[:, [2,3,4,5]]
norm = normalizeData(cols.astype(int))

x = np.delete(x, [0,1,2,3,4,5], axis=1)
arr = np.concatenate((state,grade,norm,x),axis=1)
arr = arr.astype(float)
arr = np.concatenate((x0,arr), axis=1)


x_train, x_test, y_train, y_test = model_selection.train_test_split(arr,
                                                                    y,train_size=0.7,
                                                                    test_size=0.3, 
                                                                    random_state=101)

(train_data,train_targets), (test_data,test_targets) = (x_train, y_train),(x_test, y_test)


model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape = (len(train_data[0]),)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation = "softplus"))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


epoch = 100

history = model.fit(train_data, train_targets, epochs=epoch, batch_size=10)  
mae_history = history.history['mae']      
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)


plt.plot(range(0, epoch), mae_history, 'r')
plt.title("Mean Absolute Error")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

prediction = model.predict(x_test)
mae_acc = 0
for i in range(len(prediction)):
   mae_acc = mae_acc + abs(y_test[i]-prediction[i])
mae_acc = mae_acc/len(y_test)


model2 = model_selection.KFold(n_splits = 5, random_state = 101, shuffle = True)
for i,j in model2.split(arr):
    x_train, x_test, y_train, y_test = arr[i], arr[j], y[i], y[j]
    historyval = model.fit(x_train, y_train, epochs = epoch, validation_data =(x_test,y_test))


epochs = np.arange(epoch)
acc = historyval.history['mae']
val_acc = historyval.history['val_mae']
loss = historyval.history['loss']
val_loss = historyval.history['val_loss']

fig, ax = plt.subplots()

ax.plot(epochs, loss, 'r', label='Training Loss')
ax.plot(epochs, val_loss, 'b', label='Validation Loss')
ax.set(xlabel='Epochs', ylabel='Loss',
       title='Training and Validation Loss');

ax.legend()


fig1, ax1 = plt.subplots()
ax1.plot(epochs, acc, 'r', label='Training Accuracy')
ax1.plot(epochs, val_acc, 'b', label='Validation Accuracy')
ax1.set(xlabel='Epochs', ylabel='Accuracy',
       title='Training and Aalidation Accuracy');

ax1.legend()


print()
print()

print("MAE:", test_mae_score)
print("Verified MAE:", mae_acc)

print()
print("MSE:", test_mse_score)

print()
accuracy = round(100-100*test_mae_score, 2)
print("Accuracy: ",accuracy,"%")

