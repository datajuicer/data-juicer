__version__ = "1.4.5"

import sys

from loguru import logger

# allow loading truncated images for some too large images.
from PIL import ImageFile

from data_juicer.utils.common_utils import deprecated
from data_juicer.utils.resource_utils import cuda_device_count as _count_cuda
from data_juicer.utils.resource_utils import is_cuda_available as _check_cuda

ImageFile.LOAD_TRUNCATED_IMAGES = True

# For now, only INFO will be shown. Later the severity level will be changed
# when setup_logger is called to initialize the logger.
logger.remove()
logger.add(sys.stderr, level="INFO")


cuda_device_count = deprecated(
    "`data_juicer.cuda_device_count` will be deprecated, please use `from data_juicer.utils.resource_utils import cuda_device_count`"
)(_count_cuda)
is_cuda_available = deprecated(
    "`data_juicer.is_cuda_available` will be deprecated, please use `from data_juicer.utils.resource_utils import is_cuda_available`"
)(_check_cuda)
