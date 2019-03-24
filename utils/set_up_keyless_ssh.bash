#!/bin/bash

# Validate parameters
if [ $# -le 2 ]; then
    echo "ERROR: This script requires the location of the SSH key and a list of alternating usernames and hostnames." >&2
    echo "Usage: ./$0 <path_to_ssh_key> <list_alternating_username_and_hostnames>." >&2
    echo "Example: ./$0 ~/keys/hpc_key.pem john 192.168.1.89 jane 192.168.1.90" >&2
    
    exit 1
fi

if [ ! -f $1 ]; then
    echo "ERROR: This script requires the location of the SSH key as its first parameter." >&2
    exit 1        
fi


# Run keygen command to get new SSH key if not yet there
cd `dirname $0`
HOME_DIR=`pwd`

if [ ! -d $HOME_DIR/.ssh ]; then
    mkdir $HOME_DIR/.ssh
fi

if [ ! -f $HOME_DIR/.ssh/id_rsa ]; then
    echo -e "\ny\n" | ssh-keygen -t rsa -N "" > /dev/null 2>&1
fi


# Add public key to remote hosts if keyless SSH not already working
SSH_KEY_PATH=$1
TMP_USERNAME=""
TIMEOUT_SECS=3

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=$TIMEOUT_SECS"
for param in "${@:2}"; do
    if [ "$TMP_USERNAME" == "" ]; then 
        TMP_USERNAME=$param
    else
        TMP_HOST=$param
        
        ssh $SSH_OPTS $TMP_USERNAME@$TMP_HOST ls > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            ssh $SSH_OPTS -i $SSH_KEY_PATH $TMP_USERNAME@$TMP_HOST mkdir -p .ssh
            cat $HOME_DIR/.ssh/id_rsa.pub | ssh $SSH_OPTS -i $SSH_KEY_PATH $TMP_USERNAME@$TMP_HOST 'cat >> .ssh/authorized_keys' 
        fi
        
        TMP_USERNAME=""
    fi
done


# Exit successfully
exit 0