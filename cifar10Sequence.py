from keras.utils import Sequence
import numpy as np
from skimage.io import imread
import os
from glob import glob
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import img_as_ubyte
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


class CIFAR10Sequence(Sequence):
    def __init__(self, directory_of_train_or_val, batch_size, augmentations, width, height):
        # Here, self.x is list of path to the images
        # and self.y are the associated classes.
        image_paths = list()
        y_class_names = list()
        for root, dirs, files in os.walk(directory_of_train_or_val, topdown=False):
            for name in dirs:
                # subdirectory paths: ex: path to apple folder
                y_class_path = os.path.join(root, name)
                glob_train_imgs = os.path.join(y_class_path, '*.jpg')
                class_img_paths = glob(glob_train_imgs)
                image_paths.extend(class_img_paths)
                # y_class_names should correspond to each image, may repeat several times
                y_class_names.extend([name] * len(class_img_paths))
        # image_paths: each element: 'path_to_class_folder/image_name.jpg'
        # y should be uint8 array of categorical labels
        label_encoder = LabelEncoder()
        # ex of y:[[2], [1], [1], [0], [2]], got from observing load_training_data()
        # to_categorical works on y value 0-numOfClasses, shape of return automatically be the same as above
        # must be np.uint8 type for dealing with images
        self.x, self.y = image_paths, to_categorical(label_encoder.fit_transform(y_class_names)).astype(np.uint8)
        self.batch_size = batch_size
        # augmentations passed in, can be composed augmentations
        self.augment = augmentations
        # need to resize all pictures to the same shape
        self.width = width
        self.height = height

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        # self.augment, defined in init
        # self.augment(image='*.jpg')
        # augmentations like randomBrightnessContrast has two branches, one is for image type uint8
        # use img_as_ubyte to convert image type
        return np.stack([
            self.augment(image=img_as_ubyte(resize(imread(x), (self.width, self.height))))['image'] for x in batch_x
        ], axis=0), np.array(batch_y)


'''
# Define callbacks
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=hparams.checkpoint_dir)
]
'''
'''
# Training
resnet_model.fit_generator(
    train_gen,
    epochs=hparams.n_epochs,
    validation_data=valid_gen,
    workers=2, use_multiprocessing=False,
    callbacks=callbacks)
'''