
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pymatgen.core.structure import Structure
from tensorflow.keras.models import load_model
import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
from copy import deepcopy
el = np.load('element_dict.npy', allow_pickle=True).item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p')
    parser.add_argument('--max_len', type=int, default=120)
    parser.add_argument('--epoch', type=int, default=100)
    params = parser.parse_args()
    max_len = params.max_len
    epoch = params.epoch
    predict_property = "average_voltage"
    df = pd.read_excel('F:/3D/CNN/test/预测data/Zn_battery.xlsx')
    mps = []
    tar = []
    for j in range(len(df)):
        name = df["id_charge"][j]
        crystal = Structure.from_file("F:/3D/CNN/test/预测data/" + name + ".cif")
        mp = []
        if len(crystal) < max_len:
            for i in range(len(crystal)):
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
    np.random.seed(1337)  # for reproducibility
    # from tensorflow.keras.datasets import mnist
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
    p1 = int(1 * len(tar))
    X_train = X_data[0:p1]
    Y_train = Y_data[0:p1]
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    X_train = X_train.astype('float32')
    print(len(X_train))
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
    import csv
    model = load_model('F:/3D/CNN/test/' + predict_property + '--Znqugaoxiugai.h5')

    train_pred = model.predict(X_train)
    name=[]
    for j in range(len(df)):
        na = df["id_charge"][j]
        name.append(na)
    with open('pre_Znqugaoxiugai_results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(("CIF Id", "Prediction"))
            for cif_id, pred in zip(name,train_pred):
                writer.writerow((cif_id,pred))

