echo "---------------------------"


python3 -u -W ignore train.py /home/simon/dlmi_project/pytorch-image-models/images --model=resnet34d --num-classes=2 --train-split=train_split --val-split=val_split --batch-size=32 --pretrained --epochs 20 --input-size 3 336 336 --lr=0.01 --drop-path=0.05 
