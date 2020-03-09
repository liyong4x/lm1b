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
from .postprocessor import PostprocessorWithSpecificTargets
from ..representation import (
    BrainTumorSegmentationAnnotation,
    BrainTumorSegmentationPrediction,
    SegmentationAnnotation,
    SegmentationPrediction
)
from ..config import NumberField, ConfigError
from ..preprocessor import Crop3D
from ..utils import get_size_3d_from_config


class CropSegmentationMask(PostprocessorWithSpecificTargets):
    __provider__ = 'crop_segmentation_mask'

    annotation_types = (BrainTumorSegmentationAnnotation, SegmentationAnnotation)
    prediction_types = (BrainTumorSegmentationPrediction, SegmentationPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for mask cropping"
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for mask cropping."
            ),
            'dst_volume': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination volume for mask cropping."
            ),
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size for mask cropping for both dimensions."
            )
        })
        return parameters

    def configure(self):
        self.dst_height, self.dst_width, self.dst_volume = get_size_3d_from_config(self.config, allow_none=True)
        if self.dst_height is None:
            raise ConfigError('Cropping segmentation mask require dst_height')
        if self.dst_width is None:
            ConfigError('Cropping segmentation mask require dst_width')

    def process_image(self, annotation, prediction):
        for target in annotation:
            if isinstance(target, BrainTumorSegmentationAnnotation):
                if self.dst_volume is None:
                    raise ConfigError('Cropping 3D mask require dst_volume')
                target.mask = Crop3D.crop_center(target.mask, self.dst_height, self.dst_width, self.dst_volume)
            else:
                target.mask = self.crop2d_segmentation(target.mask, self.dst_height, self.dst_width)

        for target in prediction:
            if isinstance(target, BrainTumorSegmentationPrediction):
                if self.dst_volume is None:
                    raise ConfigError('Cropping 3D mask require dst_volume')
                target.mask = Crop3D.crop_center(target.mask, self.dst_height, self.dst_width, self.dst_volume)
            else:
                target.mask = self.crop2d_segmentation(target.mask, self.dst_height, self.dst_width)

        return annotation, prediction

    @staticmethod
    def crop2d_segmentation(mask, dst_height, dst_width):
        mask_shape = np.shape(mask)
        is_hwc = len(mask_shape) == 3 and mask_shape[-1] in [1, 3]
        is_hw = len(np.shape(mask)) == 2
        if is_hw:
            height, width = mask_shape
        elif is_hwc:
            height, width = mask_shape[:2]
        else:
            height, width = mask_shape[-2:]
        diff_h = (height - dst_height) // 2
        diff_w = (width - dst_width) // 2
        if is_hw or is_hwc:
            return mask[diff_h: dst_height + diff_h, diff_w: dst_width + diff_w]
        else:
            return mask[:, diff_h: dst_height + diff_h, diff_w: dst_width + diff_w]
