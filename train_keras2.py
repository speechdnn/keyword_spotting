from sklearn.model_selection import train_test_split
import numpy as np
import keras,os
from keras.callbacks import History,ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt

freq, time_step = 128, 32
X_train=np.load('./data/x_train2.npy').reshape(-1,time_step , freq,1)
y_train=keras.utils.to_categorical(np.load("./data/y_train2.npy"),num_classes=2)
x_test=np.load("./data/x_val.npy").reshape(-1,time_step , freq,1)
y_test=np.load("./data/y_val.npy")


history = History()
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (2, 2), activation='relu', padding='valid'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(2, activation='softmax'))
model.summary()

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy'])#
filepath='keras_{epoch:04d}_{val_acc:.4f}.hdf5'
save_dir='./data'
tensorboard=TensorBoard(log_dir='./ckpt/')
ckpt=ModelCheckpoint(os.path.join(save_dir,filepath),monitor='val_acc',
                     verbose=1,save_best_only=False)#只能根据验证集进行存储,验证集不能设为validation_split=0.00
reduceLR=ReduceLROnPlateau(monitor='acc', factor=0.5,
                           patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-9)
model.fit(X_train, y_train, batch_size=64, epochs=1000,callbacks=[history,ckpt,reduceLR],validation_split=0.06)
model.save('./ckpt/kws_end.h5')
score,acc = model.evaluate(x_test, y_test, batch_size=256,verbose=1)
print('Test score:', score)
print('Test accuracy:', acc)

plt.figure()
plt.subplot(211)
plt.plot(history.history['loss'])
plt.title("Model's Training Loss")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.subplot(212)
plt.plot(history.history['acc'])
plt.title("Model's Training Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epochs')

plt.show()
