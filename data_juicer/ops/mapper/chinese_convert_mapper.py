from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

opencc = LazyLoader("opencc")

OP_NAME = "chinese_convert_mapper"

OPENCC_CONVERTER = None


def prepare_converter(mode):
    mode_path = mode + ".json"
    global OPENCC_CONVERTER
    if OPENCC_CONVERTER is None:
        # empty converter
        OPENCC_CONVERTER = opencc.OpenCC(mode_path)
    if not OPENCC_CONVERTER.config.endswith(mode_path):
        # the config is actually a config path
        # update and get a new converter with specified mode
        OPENCC_CONVERTER = opencc.OpenCC(mode_path)


@OPERATORS.register_module(OP_NAME)
class ChineseConvertMapper(Mapper):
    """Mapper to convert Chinese text between Traditional, Simplified, and Japanese Kanji.

    This operator converts Chinese text based on the specified mode. It supports conversions
    between Simplified Chinese, Traditional Chinese (including Taiwan and Hong Kong
    variants), and Japanese Kanji. The conversion is performed using a pre-defined set of
    rules. The available modes include 's2t' for Simplified to Traditional, 't2s' for
    Traditional to Simplified, and other specific variants like 's2tw', 'tw2s', 's2hk',
    'hk2s', 's2twp', 'tw2sp', 't2tw', 'tw2t', 'hk2t', 't2hk', 't2jp', and 'jp2t'. The
    operator processes text in batches and applies the conversion to the specified text key
    in the samples."""

    _batched_op = True

    def __init__(self, mode: str = "s2t", *args, **kwargs):
        """
        Initialization method.

        :param mode: Choose the mode to convert Chinese:

            s2t: Simplified Chinese to Traditional Chinese,

            t2s: Traditional Chinese to Simplified Chinese,

            s2tw: Simplified Chinese to Traditional Chinese (Taiwan Standard),

            tw2s: Traditional Chinese (Taiwan Standard) to Simplified Chinese,

            s2hk: Simplified Chinese to Traditional Chinese
            (Hong Kong variant),

            hk2s: Traditional Chinese (Hong Kong variant) to Simplified
            Chinese,

            s2twp: Simplified Chinese to Traditional Chinese (Taiwan Standard)
            with Taiwanese idiom,

            tw2sp: Traditional Chinese (Taiwan Standard) to Simplified Chinese
            with Mainland Chinese idiom,

            t2tw: Traditional Chinese to Traditional Chinese (Taiwan Standard),

            tw2t: Traditional Chinese (Taiwan standard) to Traditional Chinese,

            hk2t: Traditional Chinese (Hong Kong variant) to Traditional
            Chinese,

            t2hk: Traditional Chinese to Traditional Chinese
            (Hong Kong variant),

            t2jp: Traditional Chinese Characters (Kyūjitai) to New Japanese
            Kanji,

            jp2t: New Japanese Kanji (Shinjitai) to Traditional Chinese
            Characters,

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        mode_list = [
            "s2t",
            "t2s",
            "s2tw",
            "tw2s",
            "s2hk",
            "hk2s",
            "s2twp",
            "tw2sp",
            "t2tw",
            "tw2t",
            "hk2t",
            "t2hk",
            "t2jp",
            "jp2t",
        ]
        assert mode in mode_list, "Please make sure mode is one of {}".format(mode_list)
        self.mode = mode
        prepare_converter(self.mode)

    def process_batched(self, samples):
        prepare_converter(self.mode)

        samples[self.text_key] = [OPENCC_CONVERTER.convert(text) for text in samples[self.text_key]]
        return samples
