from __future__ import division
from keras.models import Sequential 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.models import Model
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
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

dataset=pd.DataFrame(pd.read_csv('xonly.csv',header=None))
data=np.array(dataset)
#data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
X =data[1:,1:]        #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(X)
X[allnan]=0
X2= preprocessing.scale(X)

dataset=pd.DataFrame(pd.read_csv('mthetaonly.csv',header=None))
data=np.array(dataset)
#data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
mtheta=data[1:,1:]        #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(mtheta)
mtheta[allnan]=0
#mtheta= preprocessing.scale(mtheta)

dataset=pd.DataFrame(pd.read_csv('vonly.csv',header=None))
data=np.array(dataset)
#data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
v=data[1:,1:]        #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(v)
v[allnan]=0
#mtheta= preprocessing.scale(mtheta)


dataset=pd.DataFrame(pd.read_csv('othetaonly.csv',header=None))
data=np.array(dataset)
#data = sequence.pad_sequences(data, maxlen=300,padding='post',truncating='post',dtype='float64')
otheta=data[1:,1:]        #Intentionaly remove first row to aalign it with vector labels 
allnan=isnan(otheta)
otheta[allnan]=0
#mtheta= preprocessing.scale(mtheta)
a=np.array([[1,2,3],[4,5,6],[7,8,9]])

def custscale(array,down,up):

	print np.amin(array)
	print np.amax(array)
	output=(up-np.amax(array))/(down-)		

	return None
