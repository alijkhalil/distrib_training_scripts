Overview:

    This simple repository is designed to provide the scripts 
    necessary to run a distributed deep learning module on a 
    HPC cluster. Currently it is geared towards use on AWS with 
    several similar nodes/instaces in the same placement group 
    (that are all accessible using the same public SSH key); however, 
    it can also be slightly modified for use in a lot of different 
    settings.

    The main script - 'mpi_wrapper.bash' - is the only one 
    needed to execute a distrubed deep learning module (assuming 
    the module includes code from Horovod or another framework 
    for leveraging distributed training ops).


Notes:

    -Before attempting to run the 'mpi_wrapper.bash' script, users should 
        check the parameters and sequencing by calling 'mpi_wrapper.bash' 
        without any parameters
    -For the parameters passed to 'mpi_wrapper.bash' specifying the username's 
        and host IP's included in the cluster, users should never forget to 
        include the local system IP if it is part of the cluster.
    -The training script should be in the same filesystem location on every 
        node involved in the cluster
