#!/bin/bash

export THIRD_PARTY_DIR=$(cd $(dirname $0); pwd)
export HELM_DIR=${THIRD_PARTY_DIR}/helm

# install conda
conda &> /dev/null
if [ $? -ne 0 ]; then
    echo "> setup conda ..."
    CONDA_DIR=${HOME}/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
    bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b -p $CONDA_DIR
    export PATH=$CONDA_DIR/bin:$PATH
fi

# setup helm
echo "> setup helm ..."
git clone https://github.com/stanford-crfm/helm.git
cd $HELM_DIR
git reset 33ca6e62 --hard
git apply ${THIRD_PARTY_DIR}/patch/helm.diff
conda create -n crfm-helm python=3.8 pip -y
eval "$(conda shell.bash hook)"
conda activate crfm-helm
pip install -e .