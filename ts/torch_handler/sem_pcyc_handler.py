# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""
import inspect
import torch
import os
import importlib.util
import logging

from ts.torch_handler.base_handler import BaseHandler
logger = logging.getLogger(__name__)

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        #  load the model
        self.manifest = context.manifest
        properties = context.system_properties

        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )

        model_dir = properties.get("model_dir")


        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

            # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
        else:
            logger.debug("Loading torchscript model")
            self.model = self._load_torchscript_model(model_pt_path)

        self.model.to(self.device)
        self.model.eval()
        logger.debug("Model file %s loaded successfully", model_pt_path)

        #  load the model, refer 'custom handler class' above for details

    def _list_classes_from_module(self, module, parent_class=None):
        """
        Parse user defined module to get all model service classes in it.

        :param module:
        :param parent_class:
        :return: List of model service class definitions
        """

        # Parsing the module to get all defined classes
        classes = [
            cls[1]
            for cls in inspect.getmembers(
                module,
                lambda member: inspect.isclass(member)
                               and member.__module__ == module.__name__,
            )
        ]
        # filter classes that is subclass of parent_class
        if parent_class is not None:
            return [c for c in classes if issubclass(c, parent_class)]

        return classes

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        classes = [
            cls[1]
            for cls in inspect.getmembers(
                module,
                lambda member: inspect.isclass(member)
                               and member.__module__ == module.__name__,
            )
        ]
        model_class_definitions = classes
        class_size = len(model_class_definitions)
        # if len(model_class_definitions) != 1:
        #     raise ValueError(
        #         "Expected only one class as model definition. {}".format(
        #             model_class_definitions
        #         )
        #     )

        model_class = model_class_definitions[class_size-2]
        logger.debug("class : {}".format(model_class))
        state_dict = torch.load(model_pt_path)
        model = model_class()
        model.load_state_dict(state_dict)
        return model




    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
            print(preprocessed_data)
        return preprocessed_data


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model.forward(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)