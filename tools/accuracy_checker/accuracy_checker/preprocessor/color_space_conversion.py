"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import cv2
import numpy as np

try:
    import tensorflow as tf
except ImportError as import_error:
    tf = None

from .preprocessor import Preprocessor
from ..config import ListField, ConfigError


class BgrToRgb(Preprocessor):
    __provider__ = 'bgr_to_rgb'

    def process(self, image, annotation_meta=None):
        def process_data(data):
            return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        image.data = process_data(image.data) if not isinstance(image.data, list) else [
            process_data(fragment) for fragment in image.data
        ]
        return image


class BgrToGray(Preprocessor):
    __provider__ = 'bgr_to_gray'

    def process(self, image, annotation_meta=None):
        image.data = np.expand_dims(cv2.cvtColor(image.data, cv2.COLOR_BGR2GRAY).astype(np.float32), -1)
        return image

class RgbToBgr(Preprocessor):
    __provider__ = 'rgb_to_bgr'

    def process(self, image, annotation_meta=None):
        def process_data(data):
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        image.data = process_data(image.data) if not isinstance(image.data, list) else [
            process_data(fragment) for fragment in image.data
        ]
        return image


class RgbToGray(Preprocessor):
    __provider__ = 'rgb_to_gray'

    def process(self, image, annotation_meta=None):
        image.data = np.expand_dims(cv2.cvtColor(image.data, cv2.COLOR_RGB2GRAY).astype(np.float32), -1)
        return image


class TfConvertImageDType(Preprocessor):
    __provider__ = 'tf_convert_image_dtype'

    def __init__(self, config, name, input_shapes=None):
        super().__init__(config, name, input_shapes)
        if tf is None:
            raise ImportError('*tf_convert_image_dtype* operation requires TensorFlow. Please install it before usage')
        tf.enable_eager_execution()
        self.converter = tf.image.convert_image_dtype
        self.dtype = tf.float32

    def process(self, image, annotation_meta=None):
        converted_data = self.converter(image.data, dtype=self.dtype)
        image.data = converted_data.numpy()

        return image


class SwapModalitiesBrats(Preprocessor):
    __provider__ = 'swap_modalities'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'modality_order': ListField(optional=True, default=None,
                                        description="Specifies order of modality according to model input")
        })

        return parameters

    def configure(self):
        self.modal_order = self.get_value_from_config('modality_order')
        if len(self.modal_order) != 4:
            raise ConfigError('{} supports only 4 modality, but found {}'
                              .format(self.__provider__, len(self.modal_order)))
        if max(self.modal_order) != 3 or min(self.modal_order) != 0:
            raise ConfigError('Incorrect modality index found in {} for {}'
                              .format(self.modal_order, self.__provider__))
        if len(self.modal_order) != len(set(self.modal_order)):
            raise ConfigError('Incorrect modality index found in {} for {}. Indexes must be unique'
                              .format(self.modal_order, self.__provider__))

    def process(self, image, annotation_meta=None):
        if self.modal_order is not None:
            image.data = self.swap_modalities(image.data)
        return image

    def swap_modalities(self, image):
        order = self.modal_order if image.shape[0] == 4 else [0]
        image = image[order, :, :, :]
        return image
