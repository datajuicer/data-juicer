from .formatter import FORMATTERS, LocalFormatter


@FORMATTERS.register_module()
class CsvFormatter(LocalFormatter):
    """
    The class is used to load and format csv-type files.

    Default suffixes is `['.csv']`
    """

    SUFFIXES = [".csv"]

    def __init__(self, dataset_path, suffixes=None, **kwargs):
        """
        Initialization method.

        :param dataset_path: a dataset file or a dataset directory
        :param suffixes: files with specified suffixes to be processed
        :param kwargs: extra args
        """
        super().__init__(
            dataset_path=dataset_path,
            suffixes=suffixes if suffixes else self.SUFFIXES,
            type="csv",
            **kwargs,
        )
