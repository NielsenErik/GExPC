#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    src.template_matching
    ~~~~~~~~~~~~~~~~~~~~~

    This module implements template matching modules

    :copyright: (c) 2022 by Leonardo Lucio Custode.
    :license: MIT, see LICENSE for more details.
"""
import cv2
import numpy as np
from cv_modules.processing_element import CVMetaClass, CVProcessingElement


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
