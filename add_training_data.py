import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
from sklearn.model_selection import train_test_split

"""
Add data from female speakers, and scottish and english accents to train set
(Canadian accents are classified as "native" because of similarity to US accent)
"""
def add_data(orig_X_train, orig_y_train):
    data = []
    labels = []

    for example in np.load(orig_X_train, fix_imports=True):
        data.append(example)
        # print(example.shape)
    for label in np.load(orig_y_train, fix_imports=True):
        labels.append(label)

    # Female US speakers and Canadian speakers
    native_filenames_list = ["data/clb_spectrogram_array.npy",\
        "data/eey_spectrogram_array.npy",\
        "data/ljm_spectrogram_array.npy",\
        "data/lnh_spectrogram_array.npy",\
        "data/slt_spectrogram_array.npy",\
        "data/jmk_spectrogram_array.npy",\
        "data/rxr_spectrogram_array.npy"]
    # Female Indian speakers, English male, and Scottish speakers
    non_native_filenames_list = ["data/axb_spectrogram_array.npy",\
        "data/slp_spectrogram_array.npy",\
        "data/ahw_spectrogram_array.npy",\
        "data/awb_spectrogram_array.npy",\
        "data/fem_spectrogram_array.npy"]

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

    print(len(data))
    print(len(labels))

    X_train_full, _, y_train_full, _, = train_test_split(data, labels, train_size=82158)

    X_train_full = np.array(X_train_full)
    y_train_full = np.reshape(np.array(y_train_full), (X_train_full.shape[0], 1))

    print(X_train_full.shape)
    print(y_train_full.shape)

    np.save("data/X_train_full", X_train_full)
    np.save("data/y_train_full", y_train_full)

add_data("data/X_train.npy", "data/y_train.npy")

