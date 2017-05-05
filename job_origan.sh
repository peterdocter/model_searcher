#!/bin/bash
chmod -R  777 /opt/hdfs/open_mlp/run_data/output/${PBS_JOBNAME}
cd ${TMPDIR}

uname -a
date

#HYDRA_DEBUG=1 mpiexec uname -a
#lstopo-no-graphics
#$HADOOP_PREFIX/bin/hdfs dfs -rm /user/jianzhang/models/*

ps-lite/tracker/dmlc_mpi.py -n $[4 * $PBS_NUM_NODES] -s $PBS_NUM_NODES ./remote_exec.sh

