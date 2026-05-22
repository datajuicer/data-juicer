#!/bin/bash

export THIRD_PARTY_DIR=$(cd $(dirname $0); pwd)
export MEGATRON_DIR=${THIRD_PARTY_DIR}/Megatron-LM


# setup megatron
echo "> setup Megatron-LM ..."
git clone https://github.com/NVIDIA/Megatron-LM.git
cd $MEGATRON_DIR
git reset 040eac9 --hard
git apply ${THIRD_PARTY_DIR}/patch/megatron.diff
pip install flash-attn flask flask_restful jsonlines asyncio wandb sentencepiece
