from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

ftfy = LazyLoader("ftfy")

OP_NAME = "fix_unicode_mapper"


@OPERATORS.register_module(OP_NAME)
class FixUnicodeMapper(Mapper):
    """Fixes unicode errors in text samples.

    This operator corrects common unicode errors and normalizes the text to a specified
    Unicode normalization form. The default normalization form is 'NFC', but it can be set
    to 'NFKC', 'NFD', or 'NFKD' during initialization. It processes text samples in batches,
    applying the specified normalization to each sample. If an unsupported normalization
    form is provided, a ValueError is raised."""

    _batched_op = True

    def __init__(self, normalization: str = None, *args, **kwargs):
        """
        Initialization method.

        :param normalization: the specified form of Unicode
             normalization mode, which can be one of
             ['NFC', 'NFKC', 'NFD', and 'NFKD'], default 'NFC'.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if normalization and len(normalization) > 0:
            self.normalization = normalization.upper()
        else:
            self.normalization = "NFC"

        if self.normalization.upper() not in ["NFC", "NFKC", "NFD", "NFKD"]:
            raise ValueError(
                f"Normalization mode [{normalization}] is not "
                "supported. Can only be one of "
                '["NFC", "NFKC", "NFD", "NFKD"]'
            )

    def process_batched(self, samples):
        samples[self.text_key] = [
            ftfy.fix_text(text, normalization=self.normalization) for text in samples[self.text_key]
        ]
        return samples
