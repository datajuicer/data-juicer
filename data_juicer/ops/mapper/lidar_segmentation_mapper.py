from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper

mmdet3d = LazyLoader("mmdet3d")

OP_NAME = "lidar_segmentation_mapper"


@OPERATORS.register_module(OP_NAME)
class LiDARSegmentationMapper(Mapper):
    """Mapper to do segmentation from LiDAR data."""

    _batched_op = True
    _accelerator = "cuda"

    def __init__(
        self,
        model_name="cylinder3d",
        model_cfg_name="",
        model_path="",
        tag_field_name: str = MetaKeys.lidar_segmentation_tags,
        *args,
        **kwargs,
    ):
        """
        Initialization method.
        :param model_name: Name of the model used.
        :param model_cfg_name: The config name of the model used.
        :param model_path: Path of the model weight.
        :param tag_field_name: The field name to store the tags. It's
            "lidar_segmentation_tags" in default.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        self.model_name = model_name

        if self.model_name == "cylinder3d":
            self.model_cfg_name = model_cfg_name
            self.model_path = model_path
        else:
            raise NotImplementedError(f'Only support "cylinder3d" for now, but got {self.model_name}')

        self.model_key = prepare_model(
            "mmlab",
            model_cfg=self.model_cfg_name,
            model_path=self.model_path,
            task="LiDARSegmentation",
            model_name=self.model_name,
        )
        self.tag_field_name = tag_field_name

    def process_single(self, sample, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        model = get_model(self.model_key, rank, self.use_cuda())

        results = model(dict(points=sample[self.lidar_key]))
        sample[Fields.meta][self.tag_field_name] = results[0]

        return sample
