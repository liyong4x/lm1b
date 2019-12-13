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
from ..adapters import Adapter
from ..representation import SegmentationPrediction, BrainTumorSegmentationPrediction
from ..config import ConfigValidator, BoolField, NumberField


class SegmentationAdapter(Adapter):
    __provider__ = 'segmentation'
    prediction_types = (SegmentationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'make_argmax': BoolField(
                optional=True, default=False, description="Allows to apply argmax operation to output values."
            )
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.make_argmax = self.launcher_config.get('make_argmax', False)

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        for identifier, output in zip(identifiers, raw_outputs[self.output_blob]):
            if self.make_argmax:
                output = np.argmax(output, axis=0)
            result.append(SegmentationPrediction(identifier, output))

        return result

    def _extract_predictions(self, outputs_list, meta):
        if not 'tiles_shape' in (meta[-1] or {}):
            return outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list

        tiles_shapes = [meta['tiles_shape'] for meta in meta]
        restore_output = []
        offset = 0
        for _, image_tiles_shape in enumerate(tiles_shapes):
            next_offset = offset + image_tiles_shape[0] * image_tiles_shape[1]
            image_tiles = [network_output[self.output_blob] for network_output in outputs_list[offset:next_offset]]
            tiles_columns = image_tiles[::image_tiles_shape[0]]
            image = tiles_columns[0]
            for tile_column in tiles_columns[1:]:
                image = np.concatenate((image, tile_column), axis=3)
            restore_output.append(image.squeeze())
            offset = next_offset

        return {self.output_blob: restore_output}


class SegmentationOneClassAdapter(Adapter):
    __provider__ = 'segmentation_one_class'
    prediction_types = (SegmentationPrediction, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'threshold': NumberField(
                optional=True, value_type=float, min_value=0.0, default=0.5,
                description='minimal probability threshold for separating predicted class from background'
            )
        })
        return params

    def configure(self):
        self.threshold = self.get_value_from_config('threshold')

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        for identifier, output in zip(identifiers, raw_outputs[self.output_blob]):
            output = output > self.threshold
            result.append(SegmentationPrediction(identifier, output))

        return result


class BrainTumorSegmentationAdapter(Adapter):
    __provider__ = 'brain_tumor_segmentation'
    prediction_types = (BrainTumorSegmentationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'make_argmax': BoolField(
                optional=True, default=False, description="Allows to apply argmax operation to output values."
            ),
            'shift_labels': BoolField(optional=True, default=False, description='Allows to add + 1 to argmaxed labels')
        })
        return parameters


    def configure(self):
        self.argmax = self.get_value_from_config('make_argmax')
        self.shift = self.get_value_from_config('shift_labels')

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        for identifier, output in zip(identifiers, raw_outputs[self.output_blob]):
            if self.argmax:
                output = np.argmax(output, axis=0) + int(self.shift)
                output = np.expand_dims(output, axis=0)
            result.append(BrainTumorSegmentationPrediction(identifier, output))

        return result

    def _extract_predictions(self, outputs_list, meta):
        if not (meta[-1] or {}).get('multi_infer', False):
            return outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list

        output_keys = list(outputs_list[0].keys())
        output_map = {}
        for output_key in output_keys:
            output_data = [[output[output_key] for output in outputs_list]]
            output_map[output_key] = output_data
        if 'patch_indices' in meta[-1]:
            output_shape = output_map[self.output_blob][0][0].shape
            input_data_shape = meta[0]['image_size']
            patch_indices = meta[0]['patch_indices'],
            prediction_shape = [output_shape[1]] + list(input_data_shape[1:])
            whole_prediction = self.reconstruct_from_patches(output_map[self.output_blob][0], patch_indices, prediction_shape)
            output_map[self.output_blob] = [whole_prediction]

        return output_map

    def reconstruct_from_patches(self, patches, patch_indices, data_shape, default_value=0):
        data = np.ones(data_shape) * default_value
        image_shape = data_shape[-3:]
        count = np.zeros(data_shape, dtype=np.int)
        for patch, index in zip(patches, patch_indices[0]):
            image_patch_shape = patch.shape[-3:]
            if np.any(index < 0):
                fix_patch = np.asarray((index < 0) * np.abs(index), dtype=np.int)
                patch = patch[0][..., fix_patch[0]:, fix_patch[1]:, fix_patch[2]:]
                index[index < 0] = 0
            if np.any((index + image_patch_shape) >= image_shape):
                fix_patch = np.asarray(image_patch_shape - (((index + image_patch_shape) >= image_shape)
                                                            * ((index + image_patch_shape) - image_shape)), dtype=np.int)
                patch = patch[..., :fix_patch[0], :fix_patch[1], :fix_patch[2]]
            patch_index = np.zeros(data_shape, dtype=np.bool)
            patch_index[...,
                        index[0]:index[0]+patch.shape[-3],
                        index[1]:index[1]+patch.shape[-2],
                        index[2]:index[2]+patch.shape[-1]] = True
            patch_data = np.zeros(data_shape)
            patch_data[patch_index] = patch.flatten()

            new_data_index = np.logical_and(patch_index, np.logical_not(count > 0))
            data[new_data_index] = patch_data[new_data_index]

            averaged_data_index = np.logical_and(patch_index, count > 0)
            if np.any(averaged_data_index):
                data[averaged_data_index] = (data[averaged_data_index] * count[averaged_data_index] + patch_data[averaged_data_index]) / (count[averaged_data_index] + 1)
            count[patch_index] += 1
        return data
