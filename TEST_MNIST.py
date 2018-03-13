import keras.layers as l
import keras.optimizers as o
from keras.models import Sequential
from keras.utils import to_categorical

model = Sequential()
model.add(l.Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(l.Conv2D(128, (3, 3), activation='relu'))
model.add(l.MaxPooling2D(pool_size=(2, 2)))
model.add(l.Dropout(0.5))
model.add(l.Flatten())
model.add(l.Dense(256, activation='relu'))
model.add(l.Dropout(0.5))
model.add(l.Dense(10, activation='softmax'))

sgd = o.SGD(lr = 0.01,decay=5e-6)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],28,28, 1)

import numpy as np
place_list=[]
for place,i in enumerate(y_train):
    if i == 0 or i == 1:
        place_list.append(place)
two_ways_x = np.zeros((len(place_list),28,28,1))
two_ways_y = np.zeros((len(place_list)))
print(x_test.shape)
for place,i in enumerate(place_list):
    two_ways_x[place,:] = x_train[i,:]
    two_ways_y[place] = y_train[i]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
two_ways_y = to_categorical(two_ways_y,2)

model.summary()

model.fit(x_train[:2000],y_train[:2000],batch_size=128,epochs=50,verbose=1)

model.save("10_classifier")

'''
score = model.evaluate(x_train[2000:],y_train[2000:],batch_size=128,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''


model = Sequential()
model.add(l.Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(l.Conv2D(128, (3, 3), activation='relu'))
model.add(l.MaxPooling2D(pool_size=(2, 2)))
model.add(l.Dropout(0.5))
model.add(l.Conv2D(256, (7, 7), activation='relu'))
model.add(l.Dropout(0.5))
model.add(l.Flatten())
model.add(l.Dense(256, activation='relu'))
model.add(l.Dropout(0.5))
model.add(l.Dense(2, activation='softmax'))

sgd = o.SGD(lr = 0.001,decay=1e-6)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(two_ways_x,two_ways_y,batch_size=128,epochs=50,verbose=1)

model.save("2_classifier")
