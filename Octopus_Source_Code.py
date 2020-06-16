import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import layers
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from multiprocessing import Manager
from multiprocessing import Pool

def generatedata(prog,non_prog):
    for i in range(len(non_prog)):
        data.append(non_prog[i][0].T)
    prog = np.load('TP.npy')
    for i in range(len(prog)):
        data.append(prog[i][0].T)
    temp1 = np.array([0,1])
    templist= []
    for i in range(len(non_prog)):
        templist.append(temp1)
    temp2 = np.array([1,0])
    for i in range(len(prog)):
        templist.append(temp2)
    X = np.array(data)
    Y = np.array(templist)
    return X,Y


def get_features_from_list(file):
    if '.mp3' in file:
        MP3(file).delete()
        count.append([1])
        print("started songs",len(count))
        x, sr = librosa.load(file)
        mfcc = librosa.feature.mfcc(y=x, sr=sr,n_mfcc=15)
        i=0
#         n_slides = math.floor(len(mfcc[0])/10)
#         print(n_slides)
#         count1=1
        while(i+999<len(mfcc[0])):
            
            data=[]
            data.append(np.array(mfcc[:, i:i+1000]))
            if 'Nonprog' in file:
                data.append(0)
            else:
                data.append(1)
            features.append(data)
            print("Formed Songs: ", len(features))
            i=i+random.randint(10,500)
def getdata(files):
    p=Pool(45)
    p.map(get_features_from_list,files)
    if(files contains "training"):
	
        np.save("Training.npy")
     if(files contains "testing"):
	
        np.save("Testing.npy")



def build_model():
    model = Sequential()

    model.add(Conv1D(filters=15, kernel_size=5, activation='relu', input_shape=(None, 13)))
    model.add(Conv1D(filters=25, kernel_size=5, activation='relu' ))
    model.add(GRU(
            50,
            kernel_initializer='he_normal',return_sequences=True))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(keras.layers.BatchNormalization())
    model.add(GRU(
            50,
            kernel_initializer='he_normal',return_sequences=False))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(
            2, kernel_initializer='he_normal', activation='softmax'))
    start = time.time()
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

if __name__== "__main__":
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
    prog_rock_test_path = "/home/nimrod/kiran/test_set/Test Set/Prog/"
    non_prog_rock_test_path = "/home/nimrod/kiran/test_set/Test Set/Non-Prog/"
    prog_rock_test_files = [prog_rock_test_path+f for f in listdir(prog_rock_test_path)]
    non_prog_rock_test_files = [non_prog_rock_test_path+f for f in \ listdir(non_prog_rock_test_path)]
    test_files=[]
    test_files = prog_rock_test_files + non_prog_rock_test_files
    prog_rock_path = "/content/drive/My Drive/Training_Set/Progressive Rock Songs/"
    non_prog_rock_path_1 = "/content/drive/My Drive/Training_Set/NonProg/Top Of The Pops/"
    non_prog_rock_path_2 = "/content/drive/My Drive/Training_Set/NonProg/Other Songs/"
    non_prog_rock_path_3 = "/content/drive/My Drive/Training_Set/NonProg/Additional Pop Songs/"
    prog_rock_files = [prog_rock_path+f for f in listdir(prog_rock_path)]
    non_prog_rock_files_1 = [non_prog_rock_path_1+f for f in listdir(non_prog_rock_path_1)]
    non_prog_rock_files_2 = [non_prog_rock_path_2+f for f in listdir(non_prog_rock_path_2)]
    non_prog_rock_files_3 = [non_prog_rock_path_3+f for f in listdir(non_prog_rock_path_3)]
    files=[]
    files = prog_rock_files + non_prog_rock_files_1 + non_prog_rock_files_2 + non_prog_rock_files_3
    features=manager.list([])
    getdata(files)
    getdata(test_files)
    traindata = np.load('training.npy')
    testdata = np.load('testing.npy')
    X,Y = generatedata(traindata)
    X1,Y1 = generatedata(testdata)
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
    model = build_model()
    model.fit(X_train, y_train, batch_size=50, epochs=100, validation_split=0.2)
    model.save_weights("modelgru.h5")
    file_name = 'model.sav'
    pickle.dump(model,open(file_name,'wb'))
    score, accuracy = model.evaluate(X1, Y1, batch_size=32)
    print("Accuracy on test set: ", accuracy)
    Y_predicted =  model.predict(X1)
    cm = confusion_matrix(
    Y.argmax(axis=1), Y_predicted.argmax(axis=1))
    print(cm)
    
    
