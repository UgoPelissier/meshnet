#!/usr/bin/env bash

wdir=/home/upelissier/30-Code/meshnet
ckpt_path=/home/upelissier/30-Code/meshnet/logs/version_0/checkpoints/epoch=99-step=1000.ckpt

cd $wdir
clear
python main.py test --ckpt_path $ckpt_path