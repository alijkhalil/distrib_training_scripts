This simple repository is designed to provide the scripts 
necessary to run a distributed deep learning module on a 
HPC cluster. Currently it is geared towards use on AWS with 
several similar nodes/instaces in the same placement group; 
however, it can also be slightly modified for use in a lot 
of different settings.

The main script - 'mpi_wrapper.bash' - is the only one 
needed to execute a distrubed deep learning training module 
on a cluster (assuming the module uses Horovod or another 
framework for leveraging distributed training ops).
