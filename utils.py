# Layers
from keras.layers import Dense, Activation, Flatten, Dropout, Add, BatchNormalization
from keras import backend as K

# Other
import keras
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model

# Utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random, glob
import os, sys, csv
import cv2
import time, datetime


class FixedThenFinetune(keras.callbacks.Callback):
    """
    Don't use this. Need to compile model
    """
    def __init__(self, switch_epoch):
        """
        switch_epoch: the epoch to unfreeze all model layers
        """
        self.switch_epoch = switch_epoch
        self.switched = False


    def on_epoch_begin(self, epoch, logs):
        if not self.switched and epoch > self.switch_epoch:
            print("Switching from fixed to finetune. Setting all model params as trainable.")
            set_trainable(self.model, True)
            self.model.compile(self.model.optimizer, self.model.loss, metrics=self.model._compile_metrics)
            self.switched = True


def save_class_list(OUT_DIR, class_list, model_name, dataset_name):
    class_list.sort()
    with open(os.path.join(OUT_DIR, model_name + "_" + dataset_name + "_class_list.txt"),'w') as target:
        for c in class_list:
            target.write(c)
            target.write("\n")

def load_class_list(class_list_file):
    class_list = []
    with open(class_list_file, 'r') as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            class_list.append(row)
    class_list.sort()
    return class_list

# Get a list of subfolders in the directory
def get_subfolders(directory):
    subfolders = os.listdir(directory)
    subfolders.sort()
    return subfolders

# Get number of files by searching directory recursively
def get_num_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

def set_trainable(model, is_trainable):
    for layer in model.layers:
        layer.trainable = is_trainable

# Add on new FC layers with dropout for fine tuning
def build_finetune_model(base_model, dropout, fc_layers, num_classes, as_fixed_feature_extractor=True, skip_interval=0):
    if as_fixed_feature_extractor:
        set_trainable(base_model, False)

    x = base_model.output
    x = Flatten()(x)
    for i, fc in enumerate(fc_layers):
        x = Dense(fc, activation='relu')(x) # New FC layer, random init
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        if skip_interval and i % skip_interval == 0:
            if i > 0:
                x = Add()([x, previous])
            previous = x

    predictions = Dense(num_classes, activation='softmax')(x) # New softmax layer
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')
