echo "---------------------------"


python3 -u -W ignore validate.py /home/simon/dlmi_project/pytorch-image-models/images --model=resnet34d --checkpoint=/home/simon/dlmi_project/pytorch-image-models/output/train/20240111-143115-resnet34d-336/ --results-file=/home/simon/dlmi_project/pytorch-image-models/output/train/20240111-143115-resnet34d-336/val.csv --num-classes=2  --split=test_split 
