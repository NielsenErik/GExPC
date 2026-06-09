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
from cv_modules.processing_element import ProcessingElementFactory, PEFMetaClass
from cv_modules.continuous_optimization import ContinuousOptimizationMetaClass
from cv_modules.convolution import ConvolutionModule



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
        self._seed = kwargs.get("Seed", 4)

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

    def best_module(self):
        return self._make_module(self._optimizer.best[0])
    
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
