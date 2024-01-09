import cv2
import matplotlib
import numpy as np


def load_image(image_path):
    """Load image"""

    image = cv2.imread(image_path)
    image = image / 255.0

    return image


def baseline_classify(image_path):
    """Predict class of image with baseline model"""

    image = load_image(image_path)

    # predict image class

    hsv_image = matplotlib.colors.rgb_to_hsv(image)
    average_h = np.mean(hsv_image[:, :, 0])
    print(average_h)

    pred_class = 1 if average_h < 0.60139 else 0
    image_classes = ["non Bleeding", "Bleeding"]
    image_class = image_classes[pred_class]

    return image_class