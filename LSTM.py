"""# ######################Layout of LAyers: incompleted#########################################"""
from __future__ import division
from keras.models import Sequential 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.models import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding,Dense,LSTM, SimpleRNN
from keras import initializers
import pandas as pd
from sklearn.model_selection import train_test_split                #useful for splitting data into training and test sets
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from keras.preprocessing import sequence
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import *



seed=7
np.random.seed(seed)

dataset=pd.DataFrame(pd.read_csv('xonly.csv',header=None))
data=np.array(dataset)
data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
X =data[1:,1:]        #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(X)
X[allnan]=0
X= preprocessing.scale(X)

print ('Xshape is')
print X.shape

dataset=pd.DataFrame(pd.read_csv('yonly.csv',header=None))
data=np.array(dataset)
data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
Y =data[1:,1:]           #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(Y)
Y[allnan]=0
Y= preprocessing.scale(Y)
print ('Y shape is')
print Y.shape

dataset=pd.DataFrame(pd.read_csv('mthetaonly.csv',header=None))
data=np.array(dataset)
data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
mtheta =data[1:,1:]           #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(mtheta)
mtheta[allnan]=0
mtheta= preprocessing.scale(mtheta)
print ('mtheta shape shape is')
print mtheta.shape

vec=pd.DataFrame(pd.read_excel('veclength.xlsx',sheetname='Sheet1'))
vec=np.array(vec)
Labels=vec[:X.shape[0],-1:]
(m,n)=X.shape
num_features=3
#token=int(n/4)
token=30
newarray=np.empty([m,num_features*token])

for i in range(token):
    newarray[:,num_features*i]=X[:,i+8*token]
    newarray[:,num_features*i+1]=Y[:,i+8*token]
    newarray[:,num_features*i+2]=Y[:,i+8*token]
  
#newarray=np.concatenate((data[:,0:1],newarray),axis=1)
(p,q)=newarray.shape

print newarray.shape

#scaler=MinMaxScaler()

#newarray =scaler.fit_transform(newarray)


newarray_train, newarray_test, Labels_train, Labels_test=train_test_split(newarray,Labels, test_size=0.5, random_state=42 )

Results=pd.DataFrame(pd.read_excel('Results.xlsx',sheetname='Sheet1'))
i=len(Results.index)

ep=800
newarray_train = newarray_train.reshape(newarray_train.shape[0],token, num_features)
newarray_test = newarray_test.reshape(newarray_test.shape[0],token, num_features)
print('newarray_train reshaped is')
print newarray_train.shape
print ('Labels train shape')
print Labels_train.shape

for hidden_size in [10,20,30,40,50,60,70,80,90,100]:
	model=Sequential()
	model.add(LSTM(hidden_size,input_shape=(token,num_features)))

        Results.set_value(i,'inputs features',num_features)
    	Results.set_value(i,'hidden size',hidden_size)
	model.add(Dense(1,activation='sigmoid'))


	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	history=model.fit(newarray_train,Labels_train,epochs=ep,batch_size=token)

	Results.set_value(i,'epochs',ep)
    	Results.set_value(i,'batch size',token)
	scores = model.evaluate(newarray_test, Labels_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	
	predictions = model.predict(newarray_test, verbose=1)

	print ('predictions ')
	#print predictions
	print ('predictions shape is') 
	print predictions.shape
	print (' ')
	model.save('iter%i.h5'%i)
    	Results.set_value(i,'Saved model name','iter%i.h5 '%i)

    	binarizer = Binarizer(threshold=0.2).fit(predictions)
    	binary1 = binarizer.transform(predictions)
        p=precision_score(Labels_test, binary1, average='binary')
    	print ('precision with threshold 0.2')
    	print p
    	Results.set_value(i,'precision on Test set 0.2',p)

    	p=recall_score(Labels_test, binary1, average='binary')
    	print ('recall with threshhold 0.2')
    	print p
    	Results.set_value(i,'recall on test set 0.2',p)

    	binarizer = Binarizer(threshold=0.5).fit(predictions)
    	binary2 = binarizer.transform(predictions)

    	p=precision_score(Labels_test, binary2, average='binary')
    	print ('precision with threshold 0.5')
    	print p
    	Results.set_value(i,'precision on Test set 0.5',p)
    	p=recall_score(Labels_test, binary2, average='binary')
    	print ('recall with threshhold 0.5')
    	print p
    	Results.set_value(i,'recall on test set 0.5',p)

    	binarizer = Binarizer(threshold=0.7).fit(predictions)
    	binary3 = binarizer.transform(predictions)

    	p=precision_score(Labels_test, binary3, average='binary')
    	print ('precision with threshold 0.7')
    	print p
    	Results.set_value(i,'precision on Test set 0.7',p)
    	p=recall_score(Labels_test, binary3, average='binary')
    	print ('recall with threshhold 0.7')
    	print p
    	Results.set_value(i,'recall on test set 0.7',p)

    	model.summary()
    	model.get_config()
    	writer = pd.ExcelWriter('Results.xlsx')
    	Results.to_excel(writer,'Sheet1')
    	writer.save()


        # Plot loss histroy
        # summarize history for loss
	fig=plt.figure()
	plt.plot(history.history['loss'])
	#plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.show()
        fig.savefig('loss history of model %i'%i)
         
         
 	i=i+1
	model=None

