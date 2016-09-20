#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/ZFNet/solver.prototxt \
    --gpu 0 \
    2>&1 | tee models/ZFNet/train_zfnet.log