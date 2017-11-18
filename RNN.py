"""#######################Layout of LAyers: incompleted#########################################"""
from __future__ import division
from keras.models import Sequential 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.models import Model
from keras.layers import Input, Embedding,Dense,LSTM, SimpleRNN
from keras import initializers
import pandas as pd
from sklearn.model_selection import train_test_split                #useful for splitting data into training and test sets
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer

import glob
import numpy as np


seed=7
np.random.seed(seed)

dataset=pd.DataFrame(pd.read_csv('interpolatedx.csv',header=None))
data=np.array(dataset)
X =data[:,1:-1] 
print ('Xshape is')
print X.shape
dataset=pd.DataFrame(pd.read_csv('interpolatedy.csv',header=None))
data=np.array(dataset)
Y =data[:,1:-1] 
print ('Y shape is')
print Y.shape

Labels=data[:,-1:]
(m,n)=Y.shape

token=int(n)
newarray=np.empty([m,2*token])

for i in range(token):
    newarray[:,2*i]=X[:,i]
    newarray[:,2*i+1]=Y[:,i]

 
#newarray=np.concatenate((data[:,0:1],newarray),axis=1)
(p,q)=newarray.shape

print newarray.shape

newarray = preprocessing.scale(newarray)

#newarray=newarray.reshape(p,1,1)

newarray_train, newarray_test, Labels_train, Labels_test=train_test_split(newarray,Labels, test_size=0.5, random_state=42 )
hidden_units=20
newarray_train = newarray_train.reshape(newarray_train.shape[0],newarray_train.shape[1], 1)
newarray_test = newarray_test.reshape(newarray_test.shape[0],newarray_test.shape[1], 1)
model=Sequential()
model.add(SimpleRNN(hidden_units,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=newarray_train.shape[1:]))
model.add(Dense(1, activation='sigmoid'))





model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(newarray_train,Labels_train,epochs=50,batch_size=64)

#Final evaluation of the model
scores = model.evaluate(newarray_test, Labels_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predictions = model.predict(newarray_test, verbose=1)
print (' ')
print ('predictions shape is') 
print predictions.shape
print (' ')



binarizer = Binarizer(threshold=0.2).fit(predictions)
binary1 = binarizer.transform(predictions)

p=precision_score(Labels_test, binary1, average='binary')
print ('precision with threshold 0.2')
print p
    
p=recall_score(Labels_test, binary1, average='binary')
print ('recall with threshhold 0.2')
print p
binarizer = Binarizer(threshold=0.5).fit(predictions)
binary2 = binarizer.transform(predictions)

p=precision_score(Labels_test, binary2, average='binary')
print ('precision with threshold 0.5')
print p
    
p=recall_score(Labels_test, binary2, average='binary')
print ('recall with threshhold 0.5')
print p
    
binarizer = Binarizer(threshold=0.7).fit(predictions)
binary3 = binarizer.transform(predictions)

p=precision_score(Labels_test, binary3, average='binary')
print ('precision with threshold 0.7')
print p
    
p=recall_score(Labels_test, binary3, average='binary')
print ('recall with threshhold 0.7')
print p
    
