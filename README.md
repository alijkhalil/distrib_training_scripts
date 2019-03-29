Overview:

    This simple repository is designed to provide the scripts 
    necessary to run a distributed deep learning module on a 
    HPC cluster. Currently it is geared towards use on AWS with 
    several similar nodes/instaces in the same placement group 
    (that are all accessible using the same public SSH key); however, 
    it can also be slightly modified for use in a lot of different 
    settings.

    The main script - 'mpirun_wrapper.bash' - is the only one 
    needed to execute a distrubed deep learning module (assuming 
    the module includes code from Horovod or another framework 
    for leveraging distributed training ops).  
    
    Along side that though, there are sample Keras scripts (in 
    the 'sample_train_scripts' directory) for training image 
    and video recognition models in a distributed fashion.  These
    scripts may require other repositories (capable of being 
    acquired using the 'set_up.sh' script) and the NBA play-by-play 
    dataset (also capable of being downloaded with the 
    'nba_pbp_video_dataset' repo).
    


Notes:

    -Before attempting to run the 'mpirun_wrapper.bash' script, users should 
        check the parameters and sequencing by calling 'mpirun_wrapper.bash' 
        without any parameters
    -For the parameters passed to 'mpirun_wrapper.bash' specifying the username's 
        and host IP's included in the cluster, users should never forget to 
        include the local system IP if it is part of the cluster.
    -The training script should be in the same filesystem location on every 
        node involved in the cluster
    -The 'sample_train_scripts/train_event_level.py' and 'sample_train_scripts/
        train_fbf_densenet.py' script require a very high-end GPU to run them in 
        their current state.  Model parameters in the scripts can be adjusted 
        though to reduce both their system and GPU memory footprints.