echo "---------------------------"


python3 -u -W ignore train.py /home/simon/dlmi_project/pytorch-image-models/images --model=resnet10t --num-classes=14 --train-split=split_0 --val-split=split_1 --batch-size=32 --pretrained --epochs 20 --input-size 3 336 336 --lr=0.01 --decay-rate=0.05 --decay-epochs=20 --drop-path=0.05 
