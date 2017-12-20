import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from .config import *
from .utils import *
import matplotlib.pyplot as plt

def train_model(train_x, train_y, test_x, test_y):
    model_list = os.listdir(model_dir)
    model = None
    train_scores = []
    test_scores = []
    if saved_model_filename in model_list:
        print('found previous model, recovering ...')
        model = load_model(model_dir + '/' + saved_model_filename)
    else:
        print('no previous model exist, constructing new one ...')
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', input_shape=conv_input_shape))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    for i in range(iter_cnt):
        print("running ", i, " out of ", iter_cnt, " iteration ...")
        sampled_x = train_x
        sampled_y = train_y
        # print('sampled x shape: ', sampled_x.shape)
        # print('sampled y shape: ', sampled_y.shape)
        h = model.fit(x=sampled_x, y=sampled_y, batch_size=train_batch_size, epochs=epochs)
        train_scores.extend(h.history)
        if mode == 'quick':
            score = model.evaluate(test_x, test_y, batch_size=test_batch_size)
            print('current testing score: ', score)
        if i % checkpoint == 0:
            score = model.evaluate(test_x, test_y, batch_size=test_batch_size)
            print('current testing score: ', score)
            test_scores.append(score)
        if notification:
            from .notification import send_notification, compose_iteration_msg
            msg = compose_iteration_msg(i+1, iter_cnt, score)
            send_notification(msg)
        model.save(model_dir + '/' + saved_model_filename)
    score = model.evaluate(test_x, test_y, batch_size=test_batch_size)
    print('current testing score: ', score)
    plot_learning_curve(train_scores, 'train_score.png')
    plot_learning_curve(test_scores, 'test_score.png')
