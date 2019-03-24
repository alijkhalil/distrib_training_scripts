#!/bin/bash


# Error checking
HOSTFILE_NAME="../hostfile"

if [ $# -le 2 ]; then
    echo "ERROR: This script requires at least two parameters." >&2
    echo "Usage: ./$0 <num_GPUS_per_host> <list_of_min_2_hosts>" >&2
    echo "Note that localhost (or your current IP) needs to be listed if it will be used by 'mpirun'." >&2
    
    exit 1
fi

re='^[0-9]+$'
if [[ $1 =~ $re ]] ; then
    NUM_GPUs=$1
else
    echo "ERROR: The first parameter should be a number indicating the number of GPUs per host." >&2
    exit 1
fi

for ip in "${@:2}"; do
    ping -w 2 $ip > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "ERROR: Could not reach host at '$ip'." >&2
        exit 1            
    fi
done


# Output hostfile
cd `dirname $0`
rm -f $HOSTFILE_NAME
touch $HOSTFILE_NAME

for ip in "${@:2}"; do
    NEW_LINE="$ip slots=$NUM_GPUs" 
    echo $NEW_LINE >> $HOSTFILE_NAME
done


# Exit successfully
exit 0
