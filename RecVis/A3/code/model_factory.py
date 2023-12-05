"""Python file to instantite the model and the transform that goes with it."""
from model import Net, ResNet_50, ViT_Large, ConvNeXt_Large
from data import data_transforms


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet":
            return ResNet_50()
        elif self.model_name == "vit":
            return ViT_Large()
        elif self.model_name == "convnext":
            return ConvNeXt_Large()
        else:
            raise NotImplementedError("Model not implemented")

    # Method not used because transforms are set in main.py directly
    def init_transform(self):
        return data_transforms
        # if self.model_name == "basic_cnn":
        #     return data_transforms
        # else:
        #     raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
