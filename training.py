# import pdb; pdb.set_trace()
import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pymatgen.core.structure import Structure
from keras.models import load_model
import argparse
from copy import deepcopy
import pandas as pd
import numpy as np
import random
from keras import optimizers
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import keras

el = np.load('element_dict.npy', allow_pickle=True).item()
print(el)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p')
    parser.add_argument('--max_len', type=int, default=120)
    parser.add_argument('--epoch', type=int, default=200)
    params = parser.parse_args()
    max_len = params.max_len
    epoch = params.epoch
    predict_property = "average_voltage"
    # predict_property = "capacity_grav"
    # predict_property = "capacity_vol"

    df = pd.read_excel('Zn-battery_ML/Zn_battery.xlsx')
    mps = []
    tar = []
    for j in range(len(df)):
        name = df["id_charge"][j]
        crystal = Structure.from_file("Zn-battery_ML/" + name + ".cif")
        #print(crystal)
        mp = []
        if len(crystal) < max_len:
            for i in range(len(crystal)):
                b = crystal[i].frac_coords.tolist()
                a = deepcopy(el[str(crystal[i].specie)])
                a.extend(b)
                abc=[]
                angles=[]
                #print(list(crystal.lattice.abc))
                #print(list(crystal.lattice.angles))
                for i in list(crystal.lattice.abc):
                    abc.append(i/10)
                for i in list(crystal.lattice.angles):
                    angles.append(i / 90)
                #print(abc)
                #print(angles)
                a.extend(abc)
                a.extend(angles)
                mp.append(a)
            for i in range(max_len - len(crystal)):
                mp.append([0]*26)
        else:
            for i in range(max_len):
                b = crystal[i].frac_coords.tolist()
                a = deepcopy(el[str(crystal[i].specie)])
                a.extend(b)
                abc=[]
                angles=[]
                for i in list(crystal.lattice.abc):
                    abc.append(i/10)
                for i in list(crystal.lattice.angles):
                    angles.append(i / 90)
                a.extend(abc)
                a.extend(angles)
                mp.append(a)

        mps.append([mp])
        tar.append([df[predict_property][j]])



    import numpy as np
    import scipy.io as sio
    import pandas as pd
    from sklearn.metrics import mean_squared_error

    np.random.seed(1337)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
    from keras.utils import np_utils
    from tensorflow.keras import backend as K
    import matplotlib.pyplot as plt

    img_rows, img_cols = 120, 26

    X_data = np.array(mps)
    Y_data = np.array(tar)

    random.seed(123)
    random.shuffle(X_data)
    random.seed(123)
    random.shuffle(Y_data)
    p1 = int(0.8 * len(tar))
    p2 = int(0.9* len(tar))
    X_train = X_data[0:p1]
    Y_train = Y_data[0:p1]
    X_test = X_data[p1:p2]
    Y_test= Y_data[p1:p2]
    X_val = X_data[p2:]
    Y_val = Y_data[p2:]

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    print(len(X_train), len(X_test), len(X_val))
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
    from keras.utils import np_utils
    from tensorflow.keras import backend as K
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.resnet50 import ResNet50

    model = Sequential()

    model.add(Convolution2D(24, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(24, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(48, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(96, (3, 3), padding='same'))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Convolution2D(96, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Convolution2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (3, 3), padding='same'))
    # model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])  # add lr

    from tensorflow.keras.callbacks import EarlyStopping

    np.random.seed(1337)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min')

    model.fit(X_train, Y_train, batch_size=128, epochs=epoch, verbose=1, validation_data=(X_val, Y_val),callbacks=[early_stop]
               )

    model.save(predict_property + '--Znqugaoxiugai' + '.h5')
    x = []
    num = int(len(Y_train) / 2)
    for i in range(num):
        x.append(i)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    train_pred = model.predict(X_train)
    print('training mse:', mean_squared_error(Y_train, train_pred))
    print('training mae:', mean_absolute_error(Y_train, train_pred))
    ax1.plot(x, train_pred[0:num], label='predict')
    ax1.plot(x, Y_train[0:num], label='real')
    plt.legend()
    ax1.set_title('Training Result' + str(mean_absolute_error(Y_train, train_pred)))
    ax1.set_xlabel('CIF Number')
    ax1.set_ylabel(predict_property)

    #import csv
    #with open('Li-traindata.csv', 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(("CIF Id", "Target", "Prediction"))
    #    for tar, pre in zip(Y_train,train_pred):
    #        writer.writerow((tar, pre))

    x = []
    num = len(Y_test)
    for i in range(num):
        x.append(i)
    ax2 = fig.add_subplot(2, 1, 2)
    test_pred = model.predict(X_test)
    print('testing mse:', mean_squared_error(Y_test, test_pred))
    print('testing mae:', mean_absolute_error(Y_test, test_pred))
    ax2.plot(x, test_pred[0:num], label='predict')
    ax2.plot(x, Y_test[0:num], label='real')
    plt.legend()
    ax2.set_title('Testing Result' + str(mean_absolute_error(Y_test, test_pred)))
    ax2.set_xlabel('CIF Number')
    ax2.set_ylabel(predict_property)
    plt.tight_layout()
    plt.savefig(predict_property + '--Znqugaoxiugai.png', dpi=400)
    import csv
    with open('Zn-testdataxiugai.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(("Target", "Prediction"))
        for tar, pre in zip(Y_test,test_pred):
            writer.writerow((tar, pre))
