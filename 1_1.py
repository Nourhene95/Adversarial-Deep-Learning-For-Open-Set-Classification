import keras.layers as l
import keras.optimizers as o
from keras.models import Sequential
from keras.utils import to_categorical

epoch_num = 10
batch_size = 128

generator = Sequential()
generator.add(l.Dense(256,activation='relu',input_shape=(256,),name="g_1"))
generator.add(l.Dense(9216,activation='relu',name="g_2"))
generator.add(l.Dropout(0.5,name="g_3"))
generator.add(l.Reshape((6,6,256),name="g_4"))
generator.add(l.ZeroPadding2D((6,6),name="g_5"))
generator.add(l.Conv2D(128, (7, 7), activation='relu',name="g_6"))
generator.add(l.Dropout(0.5,name="g_7"))
generator.add(l.UpSampling2D((2, 2),name="g_8"))
generator.add(l.ZeroPadding2D((2,2),name="g_9"))
generator.add(l.Conv2D(64, (3, 3), activation='relu',name="g_10"))
generator.add(l.ZeroPadding2D((2,2),name="g_11"))
generator.add(l.Conv2D(1, (3, 3), activation='relu',name="g_12"))
generator.add(l.Activation('sigmoid',name="g_13"))

discriminator = Sequential()
discriminator.add(l.Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1),name="d_1"))
discriminator.add(l.Conv2D(128, (3, 3), activation='relu',name="d_2"))
discriminator.add(l.MaxPooling2D(pool_size=(2, 2),name="d_3"))
discriminator.add(l.Dropout(0.5,name="d_4"))
discriminator.add(l.Conv2D(256, (7, 7), activation='relu',name="d_5"))
discriminator.add(l.Dropout(0.5,name="d_6"))
discriminator.add(l.Flatten(name="d_7"))
discriminator.add(l.Dense(256, activation='relu',name="d_8"))
discriminator.add(l.Dropout(0.5,name="d_9"))
discriminator.add(l.Dense(2, activation='softmax',name="d_10"))

GAN = Sequential()
GAN.add(generator)
GAN.add(discriminator)

def trainable(boolean):
    for i in range(10):
        discriminator.get_layer(name="d_"+str(i+1)).trainable = boolean
    GAN.get_layer(name="sequential_2").trainable = boolean
    discriminator.compile(loss='categorical_crossentropy',optimizer=sgd_d,metrics=['accuracy'])
    GAN.compile(loss='categorical_crossentropy',optimizer=sgd_g,metrics=['accuracy'])
  

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
two_ways_y = np.ones((len(place_list)))
for place,i in enumerate(place_list):
    two_ways_x[place,:] = x_train[i,:]
place_list=[]
for place,i in enumerate(y_test):
    if i == 0 or i == 1:
        place_list.append(place)
two_ways_x_test = np.zeros((len(place_list),28,28,1))
two_ways_y_test = np.ones((len(place_list)))
for place,i in enumerate(place_list):
    two_ways_x_test[place,:] = x_test[i,:]

two_ways_y_test = to_categorical(two_ways_y_test,2)
two_ways_y = to_categorical(two_ways_y,2)

total_length = (two_ways_y.shape[0] // batch_size) * batch_size
batch_number = total_length // batch_size
two_ways_x = two_ways_x[:total_length][:]
two_ways_y = two_ways_x[:total_length][:]

sgd_g = o.SGD(lr = 0.01,decay=1e-6)
sgd_ge = o.SGD(lr = 0.01,decay=1e-6)
sgd_d = o.SGD(lr = 0.001,decay=1e-6)
generator.compile(loss='categorical_crossentropy',optimizer=sgd_ge,metrics=['accuracy'])
discriminator.compile(loss='categorical_crossentropy',optimizer=sgd_d,metrics=['accuracy'])
GAN.compile(loss='categorical_crossentropy',optimizer=sgd_g,metrics=['accuracy'])


for i in range(epoch_num):
    for j in range(2*batch_number):
        noise = np.random.normal(0,1,(batch_size//2,256))
        train_x_fake = generator.predict(noise)
        train_y = np.concatenate((np.zeros((batch_size//2)),np.ones((batch_size//2))))
        train_y = to_categorical(train_y,2)
        #print(train_x_fake.shape)
        #print(two_ways_x[j*(batch_size//2):(j+1)*(batch_size//2)][:].shape)
        train_x = np.concatenate((train_x_fake,two_ways_x[j*(batch_size//2):(j+1)*(batch_size//2)][:]))
        trainable(True)
        #discriminator.summary()
        print("DISCRIMINATOR against both")
        discriminator.fit(train_x,train_y,batch_size)
        trainable(False)
        noise = np.random.normal(0, 1, (batch_size,256))
        target = np.ones((batch_size))
        print("GENERATOR against discriminator")
        #GAN.summary()
        GAN.fit(noise,to_categorical(target,2),batch_size)
    score = discriminator.evaluate(two_ways_x_test,two_ways_y_test,batch_size=128,verbose=0)
    print('EPOCH '+str(i+1))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

GAN.save("GAN")
discriminator.save("discriminator")
        
