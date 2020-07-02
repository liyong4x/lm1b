from csv import DictWriter
from pathlib import Path
from ...dependency import ClassProvider

PROFILERS_MAPPING = {
    (
        'accuracy',
        'accuracy_per_class',
        'classification_f1-score'
    ): 'classification',
    ('character_recognition_accuracy', ): 'char_accyracy',
    ('clip_accuracy', ): 'clip_accuracy',
    (
        'metthews_correlation_coef',
        'multi_accuracy',
        'multi_recall',
        'multi_precision',
        'f1-score'
    ): 'binary_classification',
    (
        'mae',
        'mse',
        'rmse',
        'mae_on_interval',
        'mse_on_interval',
        'rmse_on_interval',
        'angle_error'
    ): 'regression',
    ('psnr', 'ssim'): 'complex_regression',
    ('normed_error', 'per_point_normed_error'): 'point_regression',
    ('segmentation_accuracy', 'mean_iou', 'mean_accuracy', 'frequency_weighted_accuracy'): 'segmentation',
    ('coco_precision', ): 'detection'
}


class MetricProfiler(ClassProvider):
    __provider_class__ = 'metric_profiler'
    fields = ['identifier', 'result']

    def __init__(self, metric_name, dump_iterations=100):
        self.report_file = '{}.csv'.format(metric_name)
        self.out_dir = Path()
        self.dump_iterations = dump_iterations
        self.storage = []

    def generate_profiling_data(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        profiling_data = self.generate_profiling_data(*args, **kwargs)
        if isinstance(profiling_data, list):
            self.storage.extend(profiling_data)
        else:
            self.storage.append(profiling_data)
        if len(self.storage) % self.dump_iterations == 0:
            self.write_result()
            self.storage = []

    def finish(self):
        if self.storage:
            self.write_result()

    def reset(self):
        self.storage = []

    def write_result(self):
        out_path = self.out_dir / self.report_file
        new_file = not out_path.exists()

        with open(str(out_path), 'a+', newline='') as f:
            writer = DictWriter(f, fieldnames=self.fields)
            if new_file:
                writer.writeheader()
            writer.writerows(self.storage)

    def set_output_dir(self, out_dir):
        self.out_dir = out_dir
        if not out_dir.exists():
            self.out_dir.mkdir(parents=True)


def create_profiler(metric_type, metric_name):
    profiler = None
    for metric_types, profiler_id in PROFILERS_MAPPING.items():
        if metric_type in metric_types:
            return MetricProfiler.provide(profiler_id, metric_name)
    return profiler
