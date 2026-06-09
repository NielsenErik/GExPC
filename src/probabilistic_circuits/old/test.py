import cv2
import numpy as np
from cv_modules import CVMetaClass, CVProcessingElement


class TemplateMatchingModule(CVProcessingElement, metaclass=CVMetaClass):
    """
    This class implements a template matching unit
    """

    def __init__(self, templates, **kwargs):
        """
        Initializes the TemplateMatchingModule

        :template: A list of 2(or 3)-D ndarrays, consisting in the templates
        :method: one of the methods provided in cv2
        """
        CVProcessingElement.__init__(self)

        self._templates = templates
        self._method = kwargs.get("method", cv2.TM_CCOEFF_NORMED)

    def get_output(self, input_):
        """
        Computes the output of the template matching

        :input_: The image on which the template has to be matched
        :returns: A set of coordinates (x, y)
        """
        output = []
        for temp in self._templates:
            matching = cv2.matchTemplate(input_, temp, self._method)
            if self._method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                _, _, loc, _ = cv2.minMaxLoc(matching)
            else:
                _, _, _, loc = cv2.minMaxLoc(matching)
            output.extend(loc)
        return np.array(output)

    @staticmethod
    def from_params(params, config):
        """
        Creates a new template from a ndarray

        :params: An ndarray
        :config: A configuration dictionary
        :returns: A TemplateMatchingModule
        """
        if len(params.shape) == 1:
            tgt_shape = config["shape"]
            params = params.reshape(tgt_shape)
        method = config.get("method", cv2.TM_CCOEFF_NORMED)
        return TemplateMatchingModule(params, method)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self._template)
    
import torch
import numpy as np
from cv_modules import CVMetaClass, CVProcessingElement


class ConvolutionModule(CVProcessingElement, metaclass=CVMetaClass):
    """
    This class implement a convolution module composed of
    several convolutionary filters.
    This module, when called, returns a list of (x, y) coordinates
    with length equal to the number of filters, where each coordinate
    is the point of the image that returned the maximum convolution score.
    """

    def __init__(self, filters, device):
        """
        Initializes a new ConvolutionModule.

        :filters: A list of filters (i.e., list (or ndarray) of 3D ndarrays)
        :device: The device where to perform the operations
        """
        CVProcessingElement.__init__(self)
        self._device = device
        self._filters = torch.Tensor(np.array(filters))
        self._filters = self._filters.to(self._device)

    def get_output(self, input_):
        """
        This method returns the output of the agent given the input

        :input_: The agent's input (i.e., a 3d ndarray)
        :returns: The agent's output, which may be either a scalar, an ndarray
            or a torch Tensor
        """
        output = []

        X = torch.Tensor(input_)
        X = X.to(self._device)
        if len(X.shape) == 3:
            X = X.view(1, *X.shape)
        conv_output = torch.nn.functional.conv2d(
            X, self._filters).detach().cpu().numpy()[0]
        im_w = X.shape[-2]

        for out in conv_output:
            argmax = np.argmax(out.flatten())
            x = argmax % im_w
            y = argmax // im_w

            output.extend([x, y])

            # Leave a hook for further customization
            self._hook(input_, x, y)

        output = np.array(output)
        return output

    def _hook(self, input_, x, y):
        """
        This method is a hook method for allowing further
        customization

        :input_: The input
        :x: The x of the most important patch
        :y: The y of the most important patch
        :returns: TODO

        """
        pass

    @staticmethod
    def _get_filter_sizes(filter_sizes, n_filters):
        if isinstance(filter_sizes, int):
            filter_sizes = [[filter_sizes] * 2] * n_filters
        return filter_sizes

    @staticmethod
    def get_n_params(config):
        """
        Computes the number of parameters that are needed
        to produce the processing element

        :config: A dictionary containing at least the following parameters:
            - n_filters: The number of filters
            - filter_sizes: A list of (width, height) for each filter
        :returns: An int
        """
        filter_sizes = ConvolutionModule._get_filter_sizes(
            config["filter_sizes"]
        )

        acc = 0
        for w, h in filter_sizes:
            acc += w * h * 3
        return acc

    @staticmethod
    def from_params(params, config):
        """
        Creates a new ConvolutionModule from a list of parameters and
        a dictionary

        :params: An ndarray of numerical values
        :config: A dictionary containing the information to build the module
        :returns: An instance of CVProcessingElement
        """
        filter_sizes = ConvolutionModule._get_filter_sizes(
            config["filter_sizes"], config["n_filters"]
        )
        device = config.get("device", "cpu")

        filters = []

        cur_i = 0
        for w, h in filter_sizes:
            new_filter = params[cur_i:cur_i + w * h * 3]
            new_filter = new_filter.reshape(3, h, w)
            filters.append(new_filter)

        return ConvolutionModule(filters, device)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self._filters)
    
from processing_element import ProcessingElement, abstractmethod


class CVProcessingElement(ProcessingElement):
    """
    This class extends ProcessingElement to have a
    special-purpose interface
    """

    @staticmethod
    def get_n_params(config):
        """
        Computes the number of parameters that are needed
        to produce the processing element

        :config: A dictionary containing the parameters
        :returns: An int
        """
        pass

    @staticmethod
    def from_params(params, config):
        """
        Creates a new object from a list of parameters and
        a dictionary

        :params: An ndarray of numerical values
        :config: A dictionary containing the information to build the module
        :returns: An instance of CVProcessingElement
        """
        pass

    # Make concrete methods
    def set_reward(self, reward):
        """
        Allows to give the reward to the agent

        :reward: A float representing the reward
        """
        pass

    def new_episode(self):
        """
        Tells the agent that a new episode has begun
        """
        pass


class CVMetaClass(type):
    _registry = {}

    def __init__(cls, clsname, bases, methods):
        super().__init__(clsname, bases, methods)
        CVMetaClass._registry[cls.__name__] = cls

    @staticmethod
    def get(class_name):
        """
        Retrieves the class associated to the string

        :class_name: The name of the class
        :returns: A class
        """
        return CVMetaClass._registry[class_name]
    
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.factories
    ~~~~~~~~~~~~~

    This module implements factories for CV Modules

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import pickle
from cv_modules.common import CVMetaClass
from processing_element import ProcessingElementFactory, PEFMetaClass
from algorithms.continuous_optimization import ContinuousOptimizationMetaClass


class CVModuleFactory(ProcessingElementFactory, metaclass=PEFMetaClass):
    """
    This class implements a factory for CV modules.
    """

    def __init__(self, **kwargs):
        """
        Initializes the factory

        :Optimizer: A dictionary containing at least
            - name: name of the optimizer
            - kwargs: params for the optimizer
        :CVModule: A dictionary containing at least
            - name: name of the CV module
            - kwargs: params for the CV module
        """
        ProcessingElementFactory.__init__(self)

        self._opt = kwargs["Optimizer"]
        self._cvm = kwargs["CVModule"]

        # Retrieve class
        self._optimizer = ContinuousOptimizationMetaClass.get(
            self._opt["class_name"]
        )
        # Init the optimizer
        self._optimizer = self._optimizer(**self._opt["kwargs"])

        self._module_type = CVMetaClass.get(self._cvm["class_name"])

    def _make_module(self, params):
        module = self._module_type.from_params(params, self._cvm["kwargs"])
        return module

    def ask_pop(self):
        """
        This method returns a whole population of solutions for the factory.
        :returns: A population of solutions.
        """
        genes = self._optimizer.ask()

        pop = list(map(self._make_module, genes))
        return pop

    def tell_pop(self, fitnesses, data=None):
        """
        This methods assigns the computed fitness for
        each individual of the population.
        """
        self._optimizer.tell(fitnesses)


class FixedCVModuleFactory(ProcessingElementFactory, metaclass=PEFMetaClass):
    """
    This class implements a utility factory that always
    returns the same module.
    """

    def __init__(self, **kwargs):
        """
        Initializes the factory

        :Path: Path to the npy file containing the params
        :CVModule: Dict containing at least:
            - class_name: the name of the class
            - kwargs: kwargs for the class
        """
        ProcessingElementFactory.__init__(self)

        self._path = kwargs["Path"]
        self._cvm = kwargs["CVModule"]

        self._module_type = CVMetaClass.get(self._cvm["class_name"])
        print(self._path)
        data = pickle.load(open(self._path, "rb"))
        self._module = self._module_type(data, **self._cvm["kwargs"])

    def ask_pop(self):
        """
        Returns the module
        :returns: A population of solutions.
        """
        return [self._module]

    def tell_pop(self, fitnesses, data=None):
        """
        Does nothing
        """
        pass