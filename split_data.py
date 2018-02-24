import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
from sklearn.model_selection import train_test_split

"""
Load data from US male and Indian male, split into train/test/dev.
"""
def split_data():
    native_filenames_list = ["data/bdl_spectrogram_array.npy", "data/aew_spectrogram_array.npy", "data/rms_spectrogram_array.npy"]
    non_native_filenames_list = ["data/ksp_spectrogram_array.npy", "data/aup_spectrogram_array.npy", "data/gka_spectrogram_array.npy"]
    
    data = []
    labels = []
    
    for file in native_filenames_list:
        print(file)
        native_X = np.load(file, fix_imports=True)
        for example in native_X:
            data.append(example)
            labels.append(1)
            
    for file in non_native_filenames_list:
        print(file)
        non_native_X = np.load(file, fix_imports=True)
        for example in non_native_X:
            data.append(example)
            labels.append(0)

    X_train, X_devandtest, y_train, y_devandtest = train_test_split(data, labels, test_size=0.3)
    X_dev, X_test, y_dev, y_test = train_test_split(X_devandtest, y_devandtest, test_size=0.5)
    
    X_train = np.array(X_train)
    X_dev = np.array(X_dev)
    X_test = np.array(X_test)
    y_train = np.reshape(np.array(y_train), (X_train.shape[0], 1))
    y_dev = np.reshape(np.array(y_dev), (X_dev.shape[0], 1))
    y_test = np.reshape(np.array(y_test), (X_test.shape[0], 1))

    return 2, X_train, y_train, X_test, y_test, X_dev, y_dev

_, X_train, y_train, X_test, y_test, X_dev, y_dev = split_data()
np.save("data/X_train", X_train)
np.save("data/X_dev", X_dev)
np.save("data/X_test", X_test)
np.save("data/y_train", y_train)
np.save("data/y_dev", y_dev)
np.save("data/y_test", y_test)