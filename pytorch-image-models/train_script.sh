echo "---------------------------"


python3 -u -W ignore train.py /home/simon/dlmi_project/pytorch-image-models/images --model=resnet34d --initial-checkpoint=/home/simon/dlmi_project/pytorch-image-models/output/train/20231220-164404-resnet34d-336/last.pth.tar --num-classes=2 --train-split=filtered_split_1 --val-split=raw_split_0 --batch-size=32 --pretrained --epochs 20 --input-size 3 336 336 --lr=0.01 --drop-path=0.05 
