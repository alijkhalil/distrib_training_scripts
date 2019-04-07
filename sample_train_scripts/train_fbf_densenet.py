# Import statements
import math, sys, os, time

import numpy as np
import tensorflow as tf
import horovod.keras as hvd

import keras
import keras.backend as K

from keras import metrics
from keras.optimizers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../")  # May need to be changed depending on where other repos are

from state_of_art_cnns.densenet import densenet as dn  # Requires 'sys.path.append' call above

from dl_utilities.callbacks import callback_utils as cb_utils  # Requires 'sys.path.append' call above
from dl_utilities.datasets import dataset_utils as ds_utils  # Requires 'sys.path.append' call above
from dl_utilities.nba_pbp import nba_pbp_utils as pbp_utils  # Requires 'sys.path.append' call above
    
    
    
    
#############  TRAINING ROUTINE FOR FRAME-LEVEL CLASSIFICATION ############
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
    pbp_dataset_location = '/home/ubuntu/nba_pbp_labels_videos'   # REPLACE WITH YOUR SPECIFIC NBA PBP DATASET PATH
    
    batch_size = 6      # Intentionally small to fit most systems
    num_epochs = int(math.ceil(40.0 / hvd.size()))
    
    dropout = 0.2
    use_fp16 = True    
    
    num_threads = 3
    queue_size = 32
            
    train_gen, valid_gen = pbp_utils.get_train_val_nba_pbp_gens(pbp_dataset_location, batch_size, 
                                                                    nthreads=num_threads, event_level_input=False,
                                                                    num_frames_per_event=175, imbalance_factor=0.7, 
                                                                    validation_split=0.8, use_video_aug=False, 
                                                                    queue_size=queue_size)

                                                                    
    # Initialize model and set model-specific variables        
    new_kwargs = dn.SMALL_KWARGS
    new_kwargs['input_shape'] = train_gen.get_input_shape()[1:]
    new_kwargs['reduction'] = 0.25
    new_kwargs['nb_layers_per_block'] = [ 4, 4, 8, 12, 18, 20, 16, 6 ]
    new_kwargs['nb_dense_block'] = 8
    new_kwargs['depth'] = 97
    new_kwargs['classes'] = train_gen.get_output_shape()[-1]
    model = dn.DenseNet(**new_kwargs)
    
    
    # Print model summary
    model.summary()
    
    
    # Set up callbacks (starting with decreasing LR)
    callbacks = [ 
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
        keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1)
    ]
        
    if hvd.rank() == 0:
        callbacks.append(ModelCheckpoint("../output_weights/fbf_densenet.h5", monitor="acc", 
                                period=int((num_epochs + (6 * hvd.size() - 1)) // (6 * hvd.size())),
                                save_best_only=False, save_weights_only=True))
                            
    callbacks.append(cb_utils.DynamicDropoutWeights(dropout))
    
    
    # Compile model and conduct training
    compression = hvd.Compression.fp16 if use_fp16 else hvd.Compression.none

    opt = keras.optimizers.Adadelta(1.0 * hvd.size())   # Adjust learning rate based on number of GPUs
    opt = hvd.DistributedOptimizer(opt, compression=compression)
    
    model.compile(loss='categorical_crossentropy',
                        optimizer=opt, 
                        metrics=['accuracy', metrics.top_k_categorical_accuracy])   
                        
    train_steps_p_epoch = train_gen.get_steps_p_epoch()
    val_steps_p_epoch = valid_gen.get_steps_p_epoch()
    hist = model.fit_generator(
                        train_gen,
                        steps_per_epoch=(train_steps_p_epoch // hvd.size()),
                        epochs=num_epochs, 
                        initial_epoch=0,
                        callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=((2 * val_steps_p_epoch) // hvd.size()))
               
               
    # Print metrics                
    print(model.metrics_names)
    print(model.evaluate(x_test, y_test, verbose=0))


    # Return successfully
    exit(0)
