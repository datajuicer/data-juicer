from .formatter import FORMATTERS, LocalFormatter


@FORMATTERS.register_module()
class TsvFormatter(LocalFormatter):
    """
    The class is used to load and format tsv-type files.

    Default suffixes is `['.tsv']`
    """

    SUFFIXES = [".tsv"]

    def __init__(self, dataset_path, suffixes=None, **kwargs):
        """
        Initialization method.

        :param dataset_path: a dataset file or a dataset directory
        :param suffixes: files with specified suffixes to be processed
        :param kwargs: extra args, e.g. `delimiter = ','`
        """
        super().__init__(
            dataset_path=dataset_path,
            suffixes=suffixes if suffixes else self.SUFFIXES,
            type="csv",
            delimiter="\t",
            **kwargs,
        )
