import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from .config import *
from .utils import *
from .notification import *

def grid_search(train_x, train_y, test_x, test_y):
    max_score = 0
    max_model = None
    max_id = "na"
    for conv_layer_cnt in conv_layer_cnts:
        for fc_layer_cnt in fc_layer_cnts:
            for neuron_cnt in neuron_cnts:
                model = Sequential()
                model.add(Conv2D(neuron_cnt, conv_kernel, activation='relu', input_shape=conv_input_shape))
                for i in range(conv_layer_cnt):
                    model.add(Conv2D(neuron_cnt, conv_kernel, activation='relu'))
                    model.add(MaxPooling2D(pool_size=conv_pool_size))
                model.add(Dropout(conv_dropout))
                model.add(Flatten())
                for i in range(fc_layer_cnt):
                    model.add(Dense(neuron_cnt, activation='relu'))
                model.add(Dropout(fc_dropout))
                model.add(Dense(output_size, activation='softmax'))
                sgd = SGD(lr=step_size, decay=decay, momentum=momentum, nesterov=True)
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                model.fit(train_x, train_y, batch_size=train_batch_size, epochs=epochs)
                score = model.evaluate(test_x, test_y, batch_size=test_batch_size)
                print('current testing score: ', score)
    print('grid search finished')

def train_model(train_x, train_y, test_x, test_y):
    model_list = os.listdir(model_dir)
    model = None
    if saved_model_filename in model_list:
        print('found previous model, recovering ...')
        model = load_model(model_dir + '/' + saved_model_filename)
    else:
        print('no previous model exist, constructing new one ...')
        model = Sequential()
        model.add(Conv2D(conv_layers[0], conv_kernel, activation='relu', input_shape=conv_input_shape))
        for conv_layer in conv_layers[1:]:
            model.add(Conv2D(conv_layer, conv_kernel, activation='relu'))
            model.add(MaxPooling2D(pool_size=conv_pool_size))
        model.add(Dropout(conv_dropout))
        model.add(Flatten())
        for fc_layer in fc_layers:
            model.add(Dense(fc_layer, activation='relu'))
        model.add(Dropout(fc_dropout))
        model.add(Dense(output_size, activation='softmax'))
        sgd = SGD(lr=step_size, decay=decay, momentum=momentum, nesterov=True)
        loss = bloss
        if train_y.shape[1] > 1:
            loss = closs
        model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
    for i in range(iter_cnt):
        sampled_x, sampled_y = random_sample(train_x, train_y, sample_size)
        print('sampled x shape: ', sampled_x.shape)
        print('sampled y shape: ', sampled_y.shape)
        model.fit(sampled_x, sampled_y, batch_size=train_batch_size, epochs=epochs)
        score = model.evaluate(test_x, test_y, batch_size=test_batch_size)
        print('current testing score: ', score)
        msg = compose_iteration_msg(i+1, iter_cnt, score)
        send_notification(msg)
        model.save(model_dir + '/' + saved_model_filename)
        model.save_weights(model_dir + '/' + saved_weights_filename)
        with open(model_dir + '/' + saved_structure_filename, 'w') as f:
            f.write(model.to_json())
