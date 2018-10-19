from getData import *
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import matplotlib as mpl
mpl.use('Agg')
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers.convolutional import Convolution1D, AveragePooling1D
from keras.callbacks import  EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import argparse


def run_CRIP(parser):
    protein = parser.protein
    # model_dir = parser.model_dir
    batch_size = parser.batch_size
    hiddensize = parser.hiddensize
    n_epochs = parser.n_epochs
    nbfilter = parser.nbfilter
    trainXeval, test_X, trainYeval, test_y = dealwithdata(protein)
    test_y = test_y[:, 1]
    kf = KFold(len(trainYeval), n_folds=5)
    aucs = []
    for train_index, eval_index in kf:
        train_X = trainXeval[train_index]
        train_y = trainYeval[train_index]
        eval_X = trainXeval[eval_index]
        eval_y = trainYeval[eval_index]
        print('configure cnn network')
        model = Sequential()
        model.add(
            Convolution1D(input_dim=21, input_length=99, nb_filter=nbfilter, filter_length=7, border_mode="valid",
                          activation="relu", subsample_length=1))
        model.add(AveragePooling1D(pool_size=5))
        model.add(Dropout(0.5))
        # model.add(LSTM(128, input_dim=102, input_length=31, return_sequences=True))
        model.add(Bidirectional(LSTM(hiddensize, return_sequences=True)))
        model.add(Flatten())
        model.add(Dense(nbfilter, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4))  # 'rmsprop'
        print('model training')
        # checkpointer = ModelCheckpoint(filepath="models/" + protein + "_bestmodel.hdf5", verbose=0, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

        model.fit(train_X, train_y, batch_size=batch_size, nb_epoch=n_epochs, verbose=0, validation_data=(eval_X, eval_y),
                  callbacks=[earlystopper])
        predictions = model.predict_proba(test_X)[:, 1]
        auc = roc_auc_score(test_y, predictions)
        aucs.append(auc)
    print("acid AUC: %.4f " % np.mean(aucs), protein)


def parse_arguments(parser):
    parser.add_argument('--protein', type=str, metavar='<data_file>', required=True, help='the protein for training model')
    parser.add_argument('--nbfilter', type=int, default=102, help='use this option for CNN convolution')
    parser.add_argument('--hiddensize', type=int, default=120, help='use this option for LSTM')
    parser.add_argument('--batch_size', type=int, default=50, help='The size of a single mini-batch (default value: 50)')
    parser.add_argument('--n_epochs', type=int, default=30, help='The number of training epochs (default value: 30)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    run_CRIP(args)