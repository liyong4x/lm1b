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

import numpy as np
import SimpleITK as sitk

from ..config import BaseField, ConfigError, NumberField
from ..preprocessor import Preprocessor
from ..utils import get_or_parse_value


class Normalize(Preprocessor):
    __provider__ = 'normalization'

    PRECOMPUTED_MEANS = {
        'imagenet': (104.00698793, 116.66876762, 122.67891434),
        'cifar10': (125.307, 122.961, 113.8575),
    }

    PRECOMPUTED_STDS = {
        'imagenet': (104.00698793, 116.66876762, 122.67891434),
        'cifar10': (125.307, 122.961, 113.8575),
    }

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'mean': BaseField(
                optional=True,
                description="Values which will be subtracted from image channels. You can specify one "
                            "value for all channels or list of comma separated channel-wise values."
            ),
            'std': BaseField(
                optional=True,
                description="Specifies values, on which pixels will be divided. You can specify one value for all "
                            "channels or list of comma separated channel-wise values."
            )
        })
        return parameters

    def configure(self):
        self.mean = get_or_parse_value(self.config.get('mean'), Normalize.PRECOMPUTED_MEANS)
        self.std = get_or_parse_value(self.config.get('std'), Normalize.PRECOMPUTED_STDS)
        if not self.mean and not self.std:
            raise ConfigError('mean or std value should be provided')

        if self.std and 0 in self.std:
            raise ConfigError('std value should not contain 0')

        if self.mean and not (len(self.mean) == 3 or len(self.mean) == 1):
            raise ConfigError('mean should be one value or comma-separated list channel-wise values')

        if self.std and not (len(self.std) == 3 or len(self.std) == 1):
            raise ConfigError('std should be one value or comma-separated list channel-wise values')

    def process(self, image, annotation_meta=None):
        def process_data(data, mean, std):
            if self.mean:
                data = data - mean
            if self.std:
                data = data / std

            return data

        image.data = process_data(image.data, self.mean, self.std) if not isinstance(image.data, list) else [
            process_data(data_fragment, self.mean, self.std) for data_fragment in image.data
        ]

        return image


class Normalize3d(Preprocessor):
    __provider__ = "normalize3d"

    def process(self, image, annotation_meta=None):
        data = self.normalize_img(image.data)
        image_list = []
        for img in data:
            image_list.append(img)
        image.data = image_list
        image.metadata['multi_infer'] = True

        return image

    @staticmethod
    def normalize_img(img):
        for channel in range(img.shape[3]):
            channel_val = img[:, :, :, channel] - np.mean(img[:, :, :, channel])
            channel_val /= np.std(img[:, :, :, channel])
            img[:, :, :, channel] = channel_val

        return img


class N4BiasFieldCorrection(Preprocessor):
    __provider__ = 'n4_bias_field_correction'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({'skip_modality': NumberField(optional=True, min_value=0, max_value=3, value_type=int)})
        return parameters

    def configure(self):
        self.skip_modality = self.get_value_from_config('skip_modality')

    def process(self, image, annotation_meta=None):
        for m_id, modality in enumerate(image.data):
            if m_id == self.skip_modality:
                continue
            m_img = sitk.GetImageFromArray(modality)
            correct_modality = sitk.N4BiasFieldCorrection(m_img, m_img > 0)
            image.data[m_id] = sitk.GetArrayFromImage(correct_modality)
        return image
