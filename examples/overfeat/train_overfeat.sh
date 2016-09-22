#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/Overfeat/Fast/solver.prototxt \
    --gpu 0 \
    2>&1 | tee models/Overfeat/Fast/train_overfeat.log