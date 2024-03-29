from PyQt6.QtWidgets import *
from PyQt6 import uic
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import pandas as pd
import cv2
import matplotlib
import numpy as np
import torch

from timm.models import create_model

from baseline.validate_baseline import g_infere_masked_s
from baseline.derive_baseline import Baseline
from baseline.helpers import BaselineHelper



class ClassificationGUI(QMainWindow):

    def __init__(self, gui_path, images_path, metadata_path):

        super(ClassificationGUI, self).__init__()
        uic.loadUi(gui_path, self)
        self.show()

        self.images_path = images_path
        self.metadata_path = metadata_path
        self.model = self.load_model()
        self.selected_image_path = QLabel("", self)
        self.selected_image_path.setVisible(False)

        self.select_sample_Button.clicked.connect(self.select_sample)
        self.baseline_classification_Button.clicked.connect(self.baseline_classification)
        self.nn_classification_Button.clicked.connect(self.nn_classification)

        self.pred_function = g_infere_masked_s

    def select_sample(self):
        """Select image sample from explorer"""

        # select file from explorer
        selected_image_path, _ = QFileDialog.getOpenFileName(self, 'Open file', self.images_path, "Image file (*.jpg)")

        if selected_image_path != "":
            # update sample_image_field
            if self.sample_image_field.pixmap().isNull():
                 self.textBrowser.deleteLater()
            label_size = self.sample_image_field.size()
            self.image = QPixmap(selected_image_path).scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.sample_image_field.setPixmap(self.image)

            # update additional image data
            self.update_add_data(selected_image_path)

            # del prediction
            self.baseline_prediction_field.clear()
            self.nn_prediction_field.clear()

            # store image path
            self.selected_image_path.setText(selected_image_path)

    def update_add_data(self, selected_image_path):
        """Update additional image data widget"""

        # get metadata for selected image path
        image_class, fname = selected_image_path.split("/")[-2:]
        image_class = image_class.replace("_","").lower()
        df = pd.read_csv(self.metadata_path, sep=";")
        df["finding_class2"] = df["finding_class"].str.replace(" ", "")
        df["finding_class2"] = df["finding_class2"].str.replace("-", "")
        df["finding_class2"] = df["finding_class2"].str.lower()
        d = df[df["filename"].str.match(fname) & df["finding_class2"].str.match(image_class)]

        # get data set type for selected image path
            # load dfs with train, val, test separation
        train_split = pd.read_csv("../csvs/splits_by_video/train_split.csv", sep=';')
        test_split = pd.read_csv("../csvs/splits_by_video/test_split.csv", sep=';')
        val_split = pd.read_csv("../csvs/splits_by_video/val_split.csv", sep=';')
            # convert current data to identifier
        split_imge_classes = {'normalcleanmucosa': 'Normal', 'ileocecalvalve': 'Ileo-cecal valve', 'pylorus': 'Pylorus', 'erosion': 'Erosion', 'reducedmucosalview': 'Reduced Mucosal View', 'angiectasia': 'Angiectasia', 'ampullaofvater': 'Ampulla,', 'erythema': 'Erythematous', 'bloodfresh': 'Blood', 'bloodhematin': 'Blood', 'foreignbody': 'Foreign Bodies', 'ulcer': 'Ulcer', 'polyp': 'Polyp', 'lymphangiectasia': 'Lymphangiectasia'}
        image_idf = fname + ',' + split_imge_classes[image_class]
            # get dataset
        in_dataset = "train set" if image_idf in set(train_split['filename,label']) else "test set" if image_idf in set(test_split['filename,label']) else "val set" if image_idf in set(val_split['filename,label']) else "None"

        # update add data field
        if not d.empty:
            data = "filename: {0}\nvideo ID: {1}\nframe number: {2}\nfinding category: {3}\nfinding class: {4}\ndata set: {5}\nx1: {6}\ny1: {7}\nx2: {8}\ny2: {9}\nx3: {10}\ny3: {11}\nx4: {12}\ny4: {13}"\
                .format(d.iloc[0,0], d.iloc[0,1], d.iloc[0,2], d.iloc[0,3], d.iloc[0,4], in_dataset, d.iloc[0,5], d.iloc[0,6], d.iloc[0,7], d.iloc[0,8], d.iloc[0,9], d.iloc[0,10], d.iloc[0,11], d.iloc[0,12])
            self.additional_sample_information_field.setPlainText(data)
        else:
            self.additional_sample_information_field.setPlainText("No additional data found")

    def baseline_classification(self):
        """Predict image class with baseline model, based on hue values"""
        
        # case handling when no image is selected
        if self.selected_image_path.text() == "":
            msg = QMessageBox() 
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("An image must be selected.")
            msg.exec()
            return
        
        # calc average hue value
        img = cv2.imread(self.selected_image_path.text())
        hsv_image = matplotlib.colors.rgb_to_hsv(img / 255.0)
        mean = Baseline.get_masked_means(hsv_image, BaselineHelper.First_try, [np.nanmean]*3)[1] # use mean saturation for prediction
        pred_cls = self.pred_function(mean)[1]
        if pred_cls == "other":
            classification = "non Bleeding"
        else:
            classification = "Bleeding"
        self.baseline_prediction_field.setPlainText(classification)

    def nn_classification(self):
        """Predict image class with neural network model"""
        
        if self.selected_image_path.text() == "":
            msg = QMessageBox() 
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("An image must be selected.")
            msg.exec()
            return

        # load image
        image = cv2.imread(self.selected_image_path.text())
        resized_image = cv2.resize(image, dsize=(288, 288), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image

        # create torch tensor
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_stddev = np.array([0.229, 0.224, 0.225])
        normalized_image = (image/255.0 - imagenet_mean) / imagenet_stddev
        
        torch_image = torch.from_numpy(normalized_image).float()
        torch_image = torch_image.unsqueeze(0)
        torch_image = torch_image.permute((0, 3, 1, 2))

        # predict image class
        output = self.model(torch_image).softmax(-1)
        output, indices = output.topk(1)

        pred_class = 1 - indices.item()
        image_classes = ["non Bleeding", "Bleeding"]
        self.nn_prediction_field.setPlainText(image_classes[pred_class])

    def load_model(self):
        """Load model"""

        model_name = "resnet18d"
        model_path = "../models/classifier_resnet18d_20240118-150641_6.pth.tar"

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