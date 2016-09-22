#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/Overfeat/Accurate/solver.prototxt \
    --gpu 0 \
    2>&1 | tee models/Overfeat/Accurate/train_overfeat.log