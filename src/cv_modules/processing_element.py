#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    experiment_launchers.processing_element
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This module implements the interface for processing modules

    :copyright: (c) 2021 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
from abc import abstractmethod


class ProcessingElement:
    """
    This interface defines a ProcessingElement, i.e., an element that
    takes an input and produces an output and composes a pipeline.
    """

    @abstractmethod
    def get_output(self, input_):
        """
        This method returns the output of the agent given the input

        :input_: The agent's input
        :returns: The agent's output, which may be either a scalar, an ndarray
            or a torch Tensor
        """
        pass

    @abstractmethod
    def set_reward(self, reward):
        """
        Allows to give the reward to the agent

        :reward: A float representing the reward
        """
        pass

    @abstractmethod
    def new_episode(self):
        """
        Tells the agent that a new episode has begun
        """
        pass


class ProcessingElementFactory:
    """
    This class defines the interface for factories of ProcessingElements.
    """

    @abstractmethod
    def ask_pop(self):
        """
        This method returns a whole population of solutions for the factory.
        :returns: A population of solutions.
        """
        pass

    @abstractmethod
    def tell_pop(self, fitnesses, data=None):
        """
        This methods assigns the computed fitness for each individual of the population.
        """
        pass


class PEFMetaClass(type):
    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        PEFMetaClass._registry[cls.__name__] = cls
        return cls

    @staticmethod
    def get(class_name):
        """
        Retrieves the class associated to the string

        :class_name: The name of the class
        :returns: A class
        """
        return PEFMetaClass._registry[class_name]