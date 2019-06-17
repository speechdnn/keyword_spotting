from sklearn.model_selection import train_test_split
import numpy as np
import keras,os,librosa
from keras.callbacks import History
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt

freq, time_step = 128, 32
##model=load_model('./ckpt/kws_2.h5')
##model=load_model('./data/keras_066_0.9963.hdf5')
model=load_model('./data/keras_0399_0.9907.hdf5')
#model=load_model('./data/keras_069_1.0000.hdf5')#其实并不行
X_test=np.load("./data/x_val.npy").reshape(-1,time_step , freq,1)
y_test=np.load("./data/y_val.npy")

ypt = model.predict_classes(X_test).astype('int')
yt=y_test.argmax(axis=1)
error_count=0
for i in range(yt.shape[0]):
      if yt[i]!=ypt[i]:#yt[i]==1:#
            print("True:",yt[i],"Predict:",ypt[i])
            error_count+=1
path="D:/tmp/speech_dataset/"
print("标准声音dog: ")
for dirpaths, dirnames, filenames in os.walk(path):
      if dirpaths[-3:]=='dog':
            for file in filenames[2000:]:
                  if '.wav' in file:
                        datafile=os.path.join(dirpaths, file)
                        audio,fs=librosa.load(datafile,sr=16000)
                        S=librosa.feature.melspectrogram(y=audio, sr=fs,n_fft=1024,
                                       hop_length=512,n_mels=128,fmax=8000)
                        feature=librosa.feature.mfcc(S=librosa.power_to_db(S),n_mfcc=128).transpose()
                        if feature.shape[0]<32:
                              feature=np.vstack((feature,np.zeros(shape=(32-feature.shape[0],feature.shape[1]))))
                        feature=feature.reshape(-1,time_step , freq,1)
                        ypred = model.predict_classes(feature).astype('int')
                        print("ypred=",ypred,"result=",ypred==1)

#采用自己的数据-自己说的dog
files='D:/python/kws/KeywordSpotting-master/data/dog'
for i in range(1,8):
      f=files+str(i)+'.mp3'
      y,sr=librosa.load(f,sr=16000)
      S=librosa.feature.melspectrogram(y=y, sr=sr,n_fft=1024,
                                       hop_length=512,n_mels=128,fmax=8000)
      feature=librosa.feature.mfcc(S=librosa.power_to_db(S),n_mfcc=128).transpose()
      if feature.shape[0]<32:
            feature=np.vstack((feature,np.zeros(shape=(32-feature.shape[0],feature.shape[1]))))
      if feature.shape[0]>32:
            feature=feature[:32]
      feature=feature.reshape(-1,time_step , freq,1)
      ypred=model.predict_classes(feature).astype('int')
      print("Result=",ypred)
      
#错误例子是否有误判
files2='C:/Users/EnjoyTest/Desktop/chow/ml2wavtest/guzheng0'
print("古筝的声音：")
for i in range(1,21):
      f=files2+str(i)+'.wav'
      y,sr=librosa.load(f,sr=16000)
      S=librosa.feature.melspectrogram(y=y, sr=sr,n_fft=1024,
                                       hop_length=512,n_mels=128,fmax=8000)
      feature=librosa.feature.mfcc(S=librosa.power_to_db(S),n_mfcc=128).transpose()
      if feature.shape[0]<32:
            feature=np.vstack((feature,np.zeros(shape=(32-feature.shape[0],feature.shape[1]))))
      if feature.shape[0]>32:
            feature=feature[:32]
      feature=feature.reshape(-1,time_step , freq,1)
      ypred=model.predict_classes(feature).astype('int')
      print("Result=",ypred)

#错误例子2随意的输入
print("随意的声音：")
files3='D:/python/kws/KeywordSpotting-master/data/no'
for i in range(4):
      f=files3+str(i)+'.mp3'
      y,sr=librosa.load(f,sr=16000)
      S=librosa.feature.melspectrogram(y=y, sr=sr,n_fft=1024,
                                       hop_length=512,n_mels=128,fmax=8000)
      feature=librosa.feature.mfcc(S=librosa.power_to_db(S),n_mfcc=128).transpose()
      if feature.shape[0]<32:
            feature=np.vstack((feature,np.zeros(shape=(32-feature.shape[0],feature.shape[1]))))
      if feature.shape[0]>32:
            feature=feature[:32]
      feature=feature.reshape(-1,time_step , freq,1)
      ypred=model.predict_classes(feature).astype('int')
      print("Result=",ypred)
