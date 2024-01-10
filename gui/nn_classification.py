import cv2
import numpy as np
import torch
from timm.models import create_model


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_stddev = np.array([0.229, 0.224, 0.225])


model = None


def load_model():
    """Load model"""

    model_name = "resnet34d"
    model_path = "models/classifier_resnet34d_20231220-173028_9.pth.tar"

    print("loading model", model_path)
    model = create_model(
        model_name,
        num_classes=2,
        in_chans=3,
        pretrained=True,
        checkpoint_path=model_path,
    )
    model.eval()

    return model


def load_image(image_path):
    """Load image"""

    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, dsize=(288, 288), interpolation=cv2.INTER_LINEAR)
    resized_image = resized_image

    return resized_image


def nn_classify(image_path):
    global model
    """Predict class of image with NN model"""

    if model == None:
        model = load_model()

    image = load_image(image_path)

    # create torch tensor

    normalized_image = (image/255.0 - imagenet_mean) / imagenet_stddev
    
    torch_image = torch.from_numpy(normalized_image).float()
    torch_image = torch_image.unsqueeze(0)
    torch_image = torch_image.permute((0, 3, 1, 2))

    # predict image class
    output = model(torch_image).softmax(-1)
    output, indices = output.topk(1)

    pred_class = 1 - indices.item()
    image_classes = ["non Bleeding", "Bleeding"]
    image_class = image_classes[pred_class]

    return image_class



if __name__ == '__main__':
    for _ in range(3):
        print(nn_classify("pytorch-image-models/images/all/ampulla_of_vater/eb0203196e284797_1157.jpg"))