import os
import csv
from random import shuffle
from datetime import datetime
#其实标签的标记规则是
#0-非关键词的数据
#1-关键词数据
#之所以是后面加1,0或者0,1是考虑label0和label1
#当是1时,表示True,也就是这个类,一句话就当成one-hot吧


def collect_wav(path, val_percent):
      start = datetime.now()
      wav_files = []
      count_0, count_1 = 0, 0
      for dirpaths, dirnames, filenames in os.walk(path):
            if dirpaths[-3:]=='dog':#keyword
                  for file in filenames[:2000]:#choose first 200 in the list
                        if '.wav' in file:
                              count_1 += 1
                              wav_files.append((os.path.join(dirpaths, file), 1))
            elif dirpaths[-1]!='_':#_background_noise_文件内的长度不好处理
                  for file in filenames[:100]:
                        if '.wav' in file:
                              count_0 += 1
                              wav_files.append((os.path.join(dirpaths, file), 0))
      print('data: {}'.format(len(wav_files)))
    
      shuffle(wav_files)
      sep_point = int(len(wav_files) * (1-val_percent))
      print("sep_points:",sep_point)
      training_data = wav_files[:sep_point]
      validation_data = wav_files[sep_point:]
      print('training data: {} validation data: {}'.format(len(training_data), len(validation_data)))
    
      with open('./data/train2.csv', 'w') as f:
            writer = csv.writer(f)
            header = ['wav_files', 'label']
            writer.writerow(header)
            writer.writerows(training_data)
        
      with open('./data/validation2.csv', 'w') as f:
            writer = csv.writer(f)
            header = ['wav_files', 'label']
            writer.writerow(header)
            writer.writerows(validation_data)#写入的数据不同
        
      end = datetime.now()
      total = count_0 + count_1
      print('label_0 : {} ({:.3f} %), label_1 : {} ({:.3f} %)'.format(count_0, count_0/total*100, count_1, count_1/total*100))
      print('Create training and validation catalogs in ./data\nDuration: {}\n'.format(str(end-start)))

if __name__ == '__main__':
      #collect_wav(path='./data/', val_percent=0.2)#第二个参数验证集百分数
      collect_wav(path='D:/tmp/speech_dataset/', val_percent=0.0)

    
