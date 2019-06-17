#直接将数据保存了
import librosa
import pandas as pd
import numpy as np

def mfcc_single(data, label0):#(32,128)
      audio,fs=librosa.load(data,sr=16000)
      S=librosa.feature.melspectrogram(y=audio, sr=fs,n_fft=1024,
                                       hop_length=512,n_mels=128,fmax=8000)
      feature=librosa.feature.mfcc(S=librosa.power_to_db(S),n_mfcc=128).transpose()
      if feature.shape[0]<32:
            feature=np.vstack((feature,np.zeros(shape=(32-feature.shape[0],feature.shape[1]))))
      label = np.asarray([label0]).astype(float)
      return feature, label
def generate_batch():
      df_train = pd.read_csv('./data/train2.csv')
      x_train=np.zeros(shape=(0,128))
      y_train=np.zeros(shape=(0,1))
      x_val = np.zeros(shape=(0,128))
      y_val = np.zeros(shape=(0,1))
      for i in range(df_train.shape[0]):
            data = df_train.iloc[i, 0]
            label = df_train.iloc[i, 1]
            feature, label = mfcc_single(data, label)
            if feature.shape[0]==32:
                  x_train=np.vstack((x_train,feature))
                  y_train=np.vstack((y_train,label))
      batch_num=x_train.shape[0]//32
      print("batch_num=",batch_num)
      np.save('./data/x_train2.npy',x_train)
      np.save('./data/y_train2.npy',y_train)
      
##      for i in range(df_val.shape[0]):
##            data = df_val.iloc[i, 0]
##            label0, label1 = df_val.iloc[i, 1], df_val.iloc[i, 2]
##            feature, label = mfcc_single(data, label0, label1)
##            if feature.shape[0]==32:
##                  x_val=np.vstack((x_val,feature))
##                  y_val=np.vstack((y_val,label))
##      num=x_val.shape[0]//32
##      print("num=",num)
##
##      np.save('./data/x_val.npy',x_val)
##      np.save('./data/y_val.npy',y_val)
      #return X_train, y_train, batch_num, X_val, y_val
generate_batch()
