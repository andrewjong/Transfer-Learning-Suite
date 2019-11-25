from __future__ import print_function

# Networks
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.preprocessing.image import ImageDataGenerator

# Layers
from keras.layers import Dense, Activation, Flatten, Dropout
from keras import backend as K

# Other
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.callbacks import CSVLogger


from keras.models import load_model

# Utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random, glob
import os, sys, csv
import cv2
import json
import time, datetime

# Files
import utils
from utils import FixedThenFinetune

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# fix cudnn errors
import keras
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
keras.backend.tensorflow_backend.set_session(session)



# For boolean input from the command line
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Command line args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', default="my_experiment", help='name to save')
parser.add_argument('--from_epoch', type=int, default=0, help='starter epoch (for logging)')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--finetune_epochs', type=int, default=10, help='Number of epochs to finetune AFTER training fixed')
parser.add_argument('--mode', type=str, default="train", help='Select "train", or "predict" mode. \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', help='path to model to continue training')
parser.add_argument('--transfer-strategy', default="fixed", help='strategy for transfer learning. "fixed" treats the pretrained model as a fixed feature extractor; "finetune" tunes all layers', choices=("fixed", "finetune"))
parser.add_argument('--dataset', type=str, default="Pets", help='Dataset you are using.')
parser.add_argument('--max_steps', type=int, help='max steps for training')
parser.add_argument('--resize_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--resize_width', type=int, default=224, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
parser.add_argument('--h_flip', action="store_true", help='add to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', action="store_true", help='add to randomly flip the image vertically for data augmentation')
parser.add_argument('--rotation', type=int, default=0, help='Degrees to randomly rotate the image for data augmentation')
parser.add_argument('--zoom', type=float, default=0.0, help='Range for random zoom')
parser.add_argument('--shear', type=float, default=0.0, help='Shear intensity in degrees')
parser.add_argument('--brightness_offset', type=float, default=0.0, help='Brightness difference')
parser.add_argument('--model', type=str, default="MobileNet", help='Your pre-trained classification model of choice')
parser.add_argument('--summarize_model', action="store_true", help='print model summary')
parser.add_argument("--num_fc", type=int, default=4, help="number of FC layers to add")
parser.add_argument("--fc_width", type=int, default=256, help="width of FC channels")
parser.add_argument('--dropout', type=float, default=1e-3, help='Dropout ratio')
parser.add_argument('--skip_interval', type=int, default=2, help='interval to add skip connection')
parser.add_argument("--optimizer", default="Adam", help="optimizer class name")
parser.add_argument("--lr", default=0.00001, type=float, help="initial learning rate")
args = parser.parse_args()


# Global settings
BATCH_SIZE = args.batch_size
WIDTH = args.resize_width
HEIGHT = args.resize_height
FC_LAYERS = [args.fc_width for _ in range(args.num_fc)]
TRAIN_DIR = args.dataset + "/train/"
VAL_DIR = args.dataset + "/val/"
OUT_DIR = os.path.join("checkpoints", args.name)

preprocessing_function = None
base_model = None

# Create directories if needed
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)
else:
    if input(OUT_DIR + " already exists, are you sure you want to overwrite? (y/N): ").lower() != "y":
        exit()

# save args
with open(os.path.join(OUT_DIR, "args.json"), "w") as f:
    json.dump(vars(args), f, sort_keys=True, indent=4)

# Prepare the model
if args.model == "VGG16":
    from keras.applications.vgg16 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "VGG19":
    from keras.applications.vgg19 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "ResNet50":
    from keras.applications.resnet50 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "InceptionV3":
    from keras.applications.inception_v3 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "Xception":
    from keras.applications.xception import preprocess_input
    preprocessing_function = preprocess_input
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "InceptionResNetV2":
    from keras.applications.inceptionresnetv2 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "MobileNet":
    from keras.applications.mobilenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet121":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet169":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet201":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "NASNetLarge":
    from keras.applications.nasnet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = NASNetLarge(weights='imagenet', include_top=True, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "NASNetMobile":
    from keras.applications.nasnet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
else:
    ValueError("The model you requested is not supported in Keras")

    
if args.mode == "train":
    print("\n***** Begin training *****")
    print("Dataset -->", args.dataset)
    print("Resize Height -->", args.resize_height)
    print("Resize Width -->", args.resize_width)
    print("Num Epochs -->", args.num_epochs)
    print("Batch Size -->", args.batch_size)

    print("Model:")
    print("\tBase Model -->", args.model)
    print("\tNumber top layers -->", args.num_fc)
    print("\tTop layer width -->", args.fc_width)
    print("\tSkip connection interval -->", args.skip_interval)
    print("\tDropout -->", args.dropout)

    print("Data Augmentation:")
    print("\tVertical Flip -->", args.v_flip)
    print("\tHorizontal Flip -->", args.h_flip)
    print("\tRotation -->", args.rotation)
    print("\tZooming -->", args.zoom)
    print("\tShear -->", args.shear)
    print("")

    # Prepare data generators
    train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocessing_function,
      rotation_range=args.rotation,
      shear_range=args.shear,
      zoom_range=args.zoom,
      horizontal_flip=args.h_flip,
      vertical_flip=args.v_flip,
      brightness_range=[1.0 - args.brightness_offset, 1.0 + args.brightness_offset],
      rescale=1./255,
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)

    validation_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)


    # Save the list of classes for prediction mode later
    class_list = utils.get_subfolders(TRAIN_DIR)
    utils.save_class_list(OUT_DIR, class_list, model_name=args.model, dataset_name=os.path.basename(args.dataset))

    optim = eval(args.optimizer)(lr=args.lr)
    if args.continue_training is not None:
        finetune_model = load_model(args.continue_training)
        if args.transfer_strategy == "finetune":
            utils.set_trainable(finetune_model, True)
    else:
        finetune_model = utils.build_finetune_model(base_model, dropout=args.dropout, fc_layers=FC_LAYERS, num_classes=len(class_list), as_fixed_feature_extractor=True if args.transfer_strategy=="fixed" else False, skip_interval=args.skip_interval)

    finetune_model.compile(optim, loss='categorical_crossentropy', metrics=['accuracy'])
    if args.summarize_model:
        finetune_model.summary()

    num_train_images = utils.get_num_files(TRAIN_DIR)
    num_val_images = utils.get_num_files(VAL_DIR)

    def lr_decay(epoch):
        if epoch%20 == 0 and epoch!=0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr/2)
            print("LR changed to {}".format(lr/2))
        return K.get_value(model.optimizer.lr)

    learning_rate_schedule = LearningRateScheduler(lr_decay)

    # setup checkpoints
    csv_logger = CSVLogger(os.path.join(OUT_DIR, 'log.csv'), append=True, separator=';')

    latest_filepath=os.path.join(OUT_DIR, args.model + "_model_latest.h5") 
    latest_checkpoint = ModelCheckpoint(latest_filepath, monitor="accuracy", verbose=1)

    best_filepath=os.path.join(OUT_DIR, args.model + "_model_best.h5") 
    best_checkpoint = ModelCheckpoint(best_filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode='max')

    change_transfer_strategy = FixedThenFinetune(args.from_epoch + args.num_epochs)

    callbacks_list = [csv_logger, change_transfer_strategy, latest_checkpoint, best_checkpoint]

    history = finetune_model.fit_generator(train_generator, initial_epoch=args.from_epoch, epochs=args.num_epochs+args.finetune_epochs, workers=8, steps_per_epoch=args.max_steps, validation_data=validation_generator, validation_steps=num_val_images // BATCH_SIZE, class_weight='auto', shuffle=True, callbacks=callbacks_list)

    utils.plot_training(history)

elif args.mode == "predict":

    if args.image is None:
        ValueError("You must pass an image path when using prediction mode.")

    # Create directories if needed
    if not os.path.isdir("%s"%("Predictions")):
        os.makedirs("%s"%("Predictions"))

    # Read in your image
    image = cv2.imread(args.image,-1)
    save_image = image
    image = np.float32(cv2.resize(image, (HEIGHT, WIDTH)))
    image = preprocessing_function(image.reshape(1, HEIGHT, WIDTH, 3))

    class_list_file = os.path.join(OUT_DIR, args.model + "_" + os.path.basename(args.dataset) + "_class_list.txt")

    class_list = utils.load_class_list(class_list_file)
    
    finetune_model = load_model(args.continue_train)

    # Run the classifier and print results
    st = time.time()

    out = finetune_model.predict(image)

    confidence = out[0]
    class_prediction = list(out[0]).index(max(out[0]))
    class_name = class_list[class_prediction]

    run_time = time.time()-st

    print("Predicted class = ", class_name)
    print("Confidence = ", confidence)
    print("Run time = ", run_time)
    cv2.imwrite("Predictions/" + class_name[0] + ".png", save_image)
