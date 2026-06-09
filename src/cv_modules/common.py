from cv_modules.processing_element import ProcessingElement, abstractmethod


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
