import keras.layers as l
import keras.optimizers as o
from keras.models import Sequential
from keras.utils import to_categorical
from PIL import Image

epoch_num = 10
batch_size = 128

generator = Sequential()
generator.add(l.Dense(256,activation='relu',input_shape=(256,),name="ge_1"))
generator.add(l.Dense(12544,activation='relu',name="ge_2"))
generator.add(l.Dropout(0.5,name="ge_3"))
generator.add(l.Reshape((7,7,256),name="ge_4"))
generator.add(l.Conv2DTranspose(128, (2, 2),strides=(2,2), activation='relu',name="ge_6"))
generator.add(l.Dropout(0.5,name="ge_7"))
generator.add(l.Conv2DTranspose(1, (2,2),strides=(2,2), activation='relu',name="ge_10"))
generator.add(l.Activation('sigmoid',name="ge_13"))

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
GAN.add(l.Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1),name="g_1"))
GAN.add(l.Conv2D(128, (3, 3), activation='relu',name="g_2"))
GAN.add(l.MaxPooling2D(pool_size=(2, 2),name="g_3"))
GAN.add(l.Dropout(0.5,name="g_4"))
GAN.add(l.Conv2D(256, (7, 7), activation='relu',name="g_5"))
GAN.add(l.Dropout(0.5,name="g_6"))
GAN.add(l.Flatten(name="g_7"))
GAN.add(l.Dense(256, activation='relu',name="g_8"))
GAN.add(l.Dropout(0.5,name="g_9"))
GAN.add(l.Dense(2, activation='softmax',name="g_10"))
for i in GAN.layers:
    if i.name[:2] == "g_":
        i.trainable = False

def maj_weights():
    for i in GAN.layers:
        if i.name[0] == "g":
            for j in discriminator.layers:
                if j.name == "d_"+i.name[2:]:
                    #print("Weight "+i.name[2:]+" reset !")
                    i.set_weights(j.get_weights())
  

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

sgd_g = o.SGD(lr = 0.001)
sgd_d = o.SGD(lr = 0.0004)
GAN.compile(loss='categorical_crossentropy',optimizer=sgd_g,metrics=['accuracy'])
discriminator.compile(loss='categorical_crossentropy',optimizer=sgd_d,metrics=['accuracy'])

generator.summary()
discriminator.summary()
GAN.summary()

for i in range(epoch_num):
    print('EPOCH '+str(i+1))
    for j in range(2*batch_number):
        print("Batch "+str(j+1)+"/"+str(2*batch_number))
        noise = np.random.normal(0,1,(batch_size//2,256))
        train_x_fake = generator.predict(noise)
        train_y = np.concatenate((np.zeros((batch_size//2)),np.ones((batch_size//2))))
        train_y = to_categorical(train_y,2)
        train_x = np.concatenate((train_x_fake,two_ways_x[j*(batch_size//2):(j+1)*(batch_size//2)][:]))
        print("DISCRIMINATOR against both")
        hist1 = discriminator.fit(train_x,train_y,batch_size,verbose=0)
        print("Accuracy : "+str(hist1.history['acc'][-1])+" ; loss : "+str(hist1.history['loss'][-1]))
        noise = np.random.normal(0, 1, (batch_size,256))
        target = np.ones((batch_size))
        print("GENERATOR against discriminator")
        hist2 = GAN.fit(noise,to_categorical(target,2),batch_size,verbose=0)
        print("Accuracy : "+str(hist2.history['acc'][-1])+" ; loss : "+str(hist2.history['loss'][-1]))
        maj_weights()
        if j == 2*batch_number-2:
            noise = np.random.normal(0, 1, (batch_size//2,256))
            train_x_fake = generator.predict(noise)
            test_x = np.concatenate((train_x_fake,two_ways_x_test[:batch_size//2][:]))
            count = 0
            prediction = discriminator.predict(test_x)
            print(prediction)
            for k in prediction:
                if k[1]==1:
                    count+=1
            print("","")
            print("POURCENTAGE DE 1 : ",str(count/batch_size))
            print("","")
    noise = np.random.normal(0, 1, (1,256))
    train_x_fake = generator.predict(noise)
    train_x_fake = np.reshape(train_x_fake,(28,28))
    img = Image.fromarray(train_x_fake,'L')
    img.save("epoch"+str(i)+".png","PNG")
    score = discriminator.evaluate(two_ways_x_test,two_ways_y_test,batch_size=128,verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

GAN.save("GAN")
discriminator.save("discriminator")
        
