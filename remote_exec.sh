#!/bin/bash
HDFS=hdfs://yz-cpu-vm001.hogpu.cc:8020/
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar 
GPU_ID=0
if [ "$DMLC_ROLE" = "worker" ] ; then
    GPU_ID=$[$PMI_RANK / $[$PMI_SIZE / 4]]
fi
LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

python controler.py \
	--train-dir		hdfs://yz-cpu-vm001.hogpu.cc/user/great_searcher/train		\
	--cv-dir		hdfs://yz-cpu-vm001.hogpu.cc/user/great_searcher/cv			\
	--model-dir		hdfs://yz-cpu-vm001.hogpu.cc/user/great_searcher/models		\
	--gpus			0,1,2,3
