#!/bin/bash

# Ensure that there are at least the minimum required number of agruments
cd `dirname $0`

if [ $# -le 6 ]; then
    echo "ERROR: This script requires at least 7 arguments." >&2
    echo "Usage: ./$0 <path_to_SSH_key> <num_GPUs_per_node> <2_or_more_username_host_combos> <path_to_training_script>" >&2
    echo "Example: ./$0 ./ssh_key.pem 2 john 192.168.1.89 jane 192.168.1.90 joe 192.168.1.91 ./trainer.py" >&2
    
    exit 1
fi

TRAIN_FILE=${@: -1}
if [ ! -f $TRAIN_FILE ]; then
    echo "ERROR: Invalid training file passed as the last parameter."
    exit 1
fi


# Get agruments
IP_ADDRS=""
USER_IP_COMBOS=""
IS_IP=0

PARAMS=( "$@" )
unset "PARAMS[${#PARAMS[@]}-1]"
for i in `seq 2 ${#PARAMS[@]}`; do
    TMP_VAR="${PARAMS[$i]}"
    echo $TMP_VAR
    if [ $IS_IP -eq 0 ]; then
        USER_IP_COMBOS="$USER_IP_COMBOS $TMP_VAR"        
        
        IS_IP=1
    else 
        IP_ADDRS="$IP_ADDRS $TMP_VAR"        
        USER_IP_COMBOS="$USER_IP_COMBOS $TMP_VAR"        
        
        IS_IP=0
    fi
done


# Create hostfile
NUM_GPUS_P_NODE=$2

bash utils/create_hostfile.bash $NUM_GPUS_P_NODE $IP_ADDRS
if [ $? -ne 0 ]; then
    exit 1
fi


# Make SSH keyless
SSH_KEY_PATH=`readlink -f $1`
HOSTFILE_NAME="hostfile" 

bash utils/set_up_keyless_ssh.bash $SSH_KEY_PATH $USER_IP_COMBOS
if [ $? -ne 0 ]; then
    rm -f $HOSTFILE_NAME
    exit 1
fi


# Perform mpirun command
NUM_HOSTS=`echo "($# - 2) / 2" | bc`
NUM_PROC=`echo "$NUM_HOSTS * $NUM_GPUS_P_NODE" | bc`

nohup mpirun -np $NUM_PROC -hostfile $HOSTFILE_NAME --mca btl_tcp_if_include eth0 \
            -mca plm_rsh_no_tree_spawn 1 -bind-to socket -map-by slot \
            -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
            -x NCCL_MIN_NRINGS=6 -x LD_LIBRARY_PATH -x PATH \
            -mca pml ob1 -mca btl ^openib -x TF_CPP_MIN_LOG_LEVEL=0 \
            python $TRAIN_FILE &
            
            
# Remove hostfile and exit successfully
echo "MPIRUN now executing training script on $NUM_PROC GPUs accross $NUM_HOSTS hosts!"
exit 0            
