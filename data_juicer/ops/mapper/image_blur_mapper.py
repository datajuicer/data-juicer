import os

import numpy as np

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = "image_blur_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageBlurMapper(Mapper):
    """Blurs images in the dataset with a specified probability and blur type.

    This operator blurs images using one of three types: mean, box, or Gaussian. The
    probability of an image being blurred is controlled by the `p` parameter. The blur
    effect is applied using a kernel with a specified radius. Blurred images are saved to a
    directory, which can be specified or defaults to the input directory. If the save
    directory is not provided, the `DJ_PRODUCED_DATA_DIR` environment variable can be used
    to set it. The operator ensures that the blur type is one of the supported options and
    that the radius is non-negative."""

    def __init__(
        self, p: float = 0.2, blur_type: str = "gaussian", radius: float = 2, save_dir: str = None, *args, **kwargs
    ):
        """
        Initialization method.

        :param p: Probability of the image being blurred.
        :param blur_type: Type of blur kernel, including
            ['mean', 'box', 'gaussian'].
        :param radius: Radius of blur kernel.
        :param save_dir: The directory where generated image files will be stored.
            If not specified, outputs will be saved in the same directory as their corresponding input files.
            This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self._init_parameters.pop("save_dir", None)
        if blur_type not in ["mean", "box", "gaussian"]:
            raise ValueError(
                f"Blur_type [{blur_type}] is not supported. " f'Can only be one of ["mean", "box", "gaussian"]. '
            )
        if radius < 0:
            raise ValueError("Radius must be >= 0. ")

        self.p = p

        from PIL import ImageFilter

        if blur_type == "mean":
            self.blur = ImageFilter.BLUR
        elif blur_type == "box":
            self.blur = ImageFilter.BoxBlur(radius)
        else:
            self.blur = ImageFilter.GaussianBlur(radius)
        self.save_dir = save_dir

    def process_single(self, sample, context=False):
        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.source_file] = []
            return sample

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.image_key]

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )
        processed = {}
        for image_key in loaded_image_keys:
            if image_key in processed:
                continue

            if self.p < np.random.rand():
                processed[image_key] = image_key
            else:
                blured_image_key = transfer_filename(image_key, OP_NAME, self.save_dir, **self._init_parameters)
                if blured_image_key != image_key:
                    # the image_key is a valid local path, we can update it
                    if not os.path.exists(blured_image_key) or blured_image_key not in images:
                        blured_image = images[image_key].convert("RGB").filter(self.blur)
                        images[blured_image_key] = blured_image
                        blured_image.save(blured_image_key)
                        if context:
                            # update context
                            sample[Fields.context][blured_image_key] = blured_image
                    processed[image_key] = blured_image_key
                else:
                    blured_image = images[image_key].convert("RGB").filter(self.blur)
                    images[image_key] = blured_image
                    processed[image_key] = image_key
                    if context:
                        # update context
                        sample[Fields.context][image_key] = blured_image

        # when the file is modified, its source file needs to be updated.
        for i, value in enumerate(loaded_image_keys):
            if sample[Fields.source_file][i] != value:
                if processed[value] != value:
                    sample[Fields.source_file][i] = value
            if self.image_bytes_key in sample and i < len(sample[self.image_bytes_key]):
                sample[self.image_bytes_key][i] = images[processed[value]].tobytes()

        sample[self.image_key] = [processed[key] for key in loaded_image_keys]
        return sample
