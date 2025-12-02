#!/bin/bash

export THIRD_PARTY_DIR=$(cd $(dirname $0); pwd)
export EASYANIMATE_DIR=${THIRD_PARTY_DIR}/EasyAnimate

# setup easyanimate
echo "> setup easyanimate ..."
# check if the repo is ready
if [ ! -d $EASYANIMATE_DIR ]; then
  git clone https://github.com/aigc-apps/EasyAnimate.git
fi

cd $EASYANIMATE_DIR

# check if the repo is a git repo
if git rev-parse --is-inside-worktree > /dev/null 2>&1; then
    # check if the scripts are ready
    if [ ! -f "train_lora.sh" ] && [ ! -f "train_lora.py" ] && [ ! -f "infer_lora.sh" ] && [ ! -f "infer_lora.py" ] && [ ! -f "easyanimate/utils/IDDIM.py" ]; then
        git reset b54412ceb0af6a06bf907e049920f18508c862f1 --hard
        git apply ${THIRD_PARTY_DIR}/patch/easyanimate.diff
    else
        echo "WARNING: Some files in the diff are already exists, please check them or clear the repo and try again!"
    fi
else
    echo "WARNING: this dir $EASYANIMATE_DIR is not a git repo!"
fi

