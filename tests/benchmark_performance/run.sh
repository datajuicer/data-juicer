#!/bin/bash

# setup wandb configs
export WANDB_BASE_URL=$1
export WANDB_API_KEY=$2

BENCH_PATH=$(cd "$(dirname "$0")"; pwd)
RELATIVE_DJ_PATH=../..
MODALITIES=("text" "image" "video" "audio")

cd $BENCH_PATH

# 1. prepare dataset
echo "Preparing benchmark dataset..."
wget -q http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/perf_bench_data/perf_bench_data.tar.gz && tar zxf perf_bench_data.tar.gz

# 2. run the benchmark
for modality in ${MODALITIES[@]}
do
    echo "Running benchmark for $modality modality..."
    python $RELATIVE_DJ_PATH/tools/process_data.py --config configs/$modality.yaml
done

# 3. collect & upload benchmark results
echo "Collecting and reporting benchmark results..."
python report.py

# 4. clear resources
echo "Clearing resources..."
rm -rf perf_bench_data.tar.gz
rm -rf perf_bench_data/
