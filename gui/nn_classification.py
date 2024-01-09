


def load_model(model_path):
    """Load model"""

    model = None

    # load model

    return model


def load_image(image_path):
    """Load image"""

    image = None
    # load image

    return image


def nn_classify(image_path, model_path):
    """Predict class of image with NN model"""

    model = load_model(model_path)
    image = load_image(image_path)
    
    # predict image class

    pred_class = 1
    image_classes = ["non Bleeding", "Bleeding"]
    image_class = image_classes[pred_class]

    return image_class