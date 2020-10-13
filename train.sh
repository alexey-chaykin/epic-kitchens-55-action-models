python train.py ucf101 RGB train.txt eval.txt --arch resnet50 --num_segments 8 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b 128 -j 0 --dropout 0.8 --snapshot_pref TSN_
