import yaml
import sys
import os

sys.path.append("./detection/fd/models/retinaface/")
from .models.retinaface.retinaface import RetinaFace


class ModelLoaderFactory:
    """
    A class used to create a detection model instance and load it to device
    ...

    Attributes
    ----------

    model_type : str
        type of model Head as in conf file (ex. RetinaFace)
    model_path : str
        path to weights.json
    model_params : dict
        dict with model hyper parameters
    supported_models : list
        list of supported model types

    Methods
    -------
    load_model(device)
        creates model instance uring self.model_params and model class determined by self.model_type
        and loads it to memory on device

    """

    def __init__(self, model_type, model_path, conf):
        """
        Parameters
        ----------

        model_type : str
            type of model Head as in conf file (ex. RetinaFace)
        model_path : str
            path to weights.json
        conf : str
            yaml config file with model hyper parameters

        """
        self.model_type = model_type
        self.model_path = model_path
        self.supported_models = ["RetinaFace"]
        print(os.getcwd())
        with open(conf) as f:
            model_conf = yaml.safe_load(f)
            self.model_params = model_conf[model_type]
        print("Detection model parameters:")
        print(self.model_params)

    def load_model(self, device):
        """
        Parameters
        ----------

        device : str
            a string representing a device to use for storing models and images, "cuda:<id>" for gpu, "cpu" for cpu
        raises:
            NameError if model_type not in supported_models
        """
        if self.model_type == "RetinaFace":
            model = RetinaFace(
                prefix=self.model_path,
                epoch=self.model_params["epoch"],
                ctx_id=int(device[-1]) if "cuda" in device else -1,
                network=self.model_params["network"],
                nms=self.model_params["nms"],
                nocrop=self.model_params["nocrop"],
                decay4=self.model_params["decay4"],
                vote=self.model_params["vote"],
                image_h=self.model_params["image_h"],
                image_w=self.model_params["image_w"],
            )
        else:
            raise NameError(
                f"model type {self.model_type} is not supported, supported models are {self.supported_models}"
            )

        return model
