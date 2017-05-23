#!/bin/bash
chmod -R  777 /opt/hdfs/open_mlp/run_data/output/${PBS_JOBNAME}
cd ${TMPDIR}

uname -a
date

#HYDRA_DEBUG=1 mpiexec uname -a
#lstopo-no-graphics
#$HADOOP_PREFIX/bin/hdfs dfs -rm /user/jianzhang/models/*

HDFS=hdfs://yz-cpu-vm001.hogpu.cc:8020/
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar 
GPU_ID=0
if [ "$DMLC_ROLE" = "worker" ] ; then
    GPU_ID=$[$PMI_RANK / $[$PMI_SIZE / 4]]
fi
LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

./lunch_thread.sh
