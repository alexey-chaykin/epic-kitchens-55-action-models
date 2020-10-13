# EPIC-KITCHENS-55 action recognition models (adapted)

This is [EPIC-KITCHENS-55 action recognition models repo](https://github.com/epic-kitchens/epic-kitchens-55-action-models) 
with some parts of original [TSN](https://github.com/yjxiong/tsn-pytorch) and [TRN](https://github.com/metalbubble/TRN-pytorch) 
projects adopted, to be able to train a new model and test it on video files.

To test pretrained models run:
```bash
$ ./test.sh pretrain/TRN_arch\=resnet50_modality\=RGB_segments\=8-c8176b38.pth.tar
$ ./test.sh pretrain/TSN_arch\=resnet50_modality\=RGB_segments\=8-3ecf904f.pth.tar
```

To train a new model on your images:
1. Extend train dataset located at train_data and edit train.txt accordingly
2. Extend eval dataset located at eval_data and edit eval.txt accordingly
3. Start training by:
```bash
$ ./train.sh
```

To test retrained model run:
```bash
$ ./test.sh TSN__rgb_checkpoint.pth.tar 
```
or
```bash
$ ./test.sh TSN__rgb_model_best.pth.tar 
```

