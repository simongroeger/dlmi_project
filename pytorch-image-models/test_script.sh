echo "---------------------------"


python3 -u -W ignore validate.py /home/simon/dlmi_project/pytorch-image-models/images/metric-csv --model=resnet18d --checkpoint=/home/simon/dlmi_project/pytorch-image-models/output/train/20240129-190333-resnet18d-336/ --results-file=/home/simon/dlmi_project/pytorch-image-models/output/train/20240129-190333-resnet18d-336/val.csv --num-classes=2  --split=test_split 
