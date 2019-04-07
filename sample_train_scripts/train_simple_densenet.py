# Import statements
import math, sys, os, time

import numpy as np
import tensorflow as tf
import horovod.keras as hvd

import keras
import keras.backend as K

from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from keras import metrics
from keras.optimizers import *
from keras.models import Model
from keras.utils import to_categorical

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../")  # May need to be changed depending on where other repos are

from state_of_art_cnns.densenet import densenet as dn  # Requires 'sys.path.append' call above
from dl_utilities.callbacks import callback_utils as cb_utils  # Requires 'sys.path.append' call above
from dl_utilities.datasets import dataset_utils as ds_utils  # Requires 'sys.path.append' call above
from dl_utilities.nba_pbp import nba_pbp_utils as pbp_utils  # Requires 'sys.path.append' call above
    
    
    
    
#############  TRAINING ROUTINE FOR DENSENET ON CIFAR100 ############
if __name__ == '__main__':
	
	# Initialize Horovod.
    hvd.init()
	
	
    # Set up TF session
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)	
    config.gpu_options.allow_growth=True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    sess = tf.Session(config=config)
    K.set_session(sess)
	
    
    # Get training/test data and normalize/standardize it    
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train, x_test = ds_utils.normal_image_preprocess(x_train, x_test)

    
	# Convert class vectors to sparse/binary class matrices
    num_classes = 100
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)


    # Set up image augmentation generator
    global_image_aug = ImageDataGenerator(
                                    rotation_range=15, 
                                    width_shift_range=(6. / x_train.shape[2]), 
                                    height_shift_range=(6. / x_train.shape[1]), 
                                    horizontal_flip=True, 
                                    zoom_range=0.175)

    
    # Initialize model        
    model = dn.DenseNet(**dn.LARGE_KWARGS)
    
    
    # Print model summary
    model.summary()
    
    
    # Set up callbacks (starting with decreasing LR)
    num_epochs = int(math.ceil(150.0 / hvd.size()))
    dropout = 0.2

    callbacks = [ 
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
        keras.callbacks.ReduceLROnPlateau(patience=6, verbose=1)
    ]
        
    if hvd.rank() == 0:
        callbacks.append(ModelCheckpoint("../output_weights/simple_densenet.h5", monitor="acc", 
                                period=int((num_epochs + (10 * hvd.size() - 1)) // (10 * hvd.size())),
                                save_best_only=False, save_weights_only=True))
                            
    callbacks.append(cb_utils.DynamicDropoutWeights(dropout))
    
    
    # Compile model and conduct training
    use_fp16 = True    
    batch_size = 128      # Intentionally small to fit most systems    

    compression = hvd.Compression.fp16 if use_fp16 else hvd.Compression.none
    
    opt = keras.optimizers.Adadelta(1.0 * hvd.size())   # Adjust learning rate based on number of GPUs
    opt = hvd.DistributedOptimizer(opt, compression=compression)
        
    model.compile(loss='categorical_crossentropy',
                        optimizer=opt, 
                        metrics=['accuracy', metrics.top_k_categorical_accuracy])   
                        
    hist = model.fit_generator(
                    global_image_aug.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=(x_train.shape[0] // batch_size),
                    epochs=num_epochs, 
                    initial_epoch=0,
                    callbacks=callbacks,
                    validation_data=(x_test, y_test))
               
               
    # Print metrics                
    print(model.metrics_names)
    print(model.evaluate(x_test, y_test, verbose=0))


    # Return successfully
    exit(0)
