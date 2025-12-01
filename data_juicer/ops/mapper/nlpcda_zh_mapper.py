from copy import deepcopy

from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.logger_utils import HiddenPrints

from ..base_op import OPERATORS, Mapper

nlpcda = LazyLoader("nlpcda")

OP_NAME = "nlpcda_zh_mapper"


@OPERATORS.register_module(OP_NAME)
class NlpcdaZhMapper(Mapper):
    """Augments Chinese text samples using the nlpcda library.

    This operator applies various augmentation methods to Chinese text, such as replacing
    similar words, homophones, deleting random characters, swapping characters, and
    replacing equivalent numbers. The number of augmented samples generated can be
    controlled by the `aug_num` parameter. If `sequential` is set to True, the augmentation
    methods are applied in sequence; otherwise, they are applied independently. The original
    sample can be kept or removed based on the `keep_original_sample` flag. It is
    recommended to use 1-3 augmentation methods at a time to avoid significant changes in
    the semantics of the samples. Some augmentation methods may not work for special texts,
    resulting in no augmented samples being generated."""

    _batched_op = True

    def __init__(
        self,
        sequential: bool = False,
        aug_num: PositiveInt = 1,
        keep_original_sample: bool = True,
        replace_similar_word: bool = False,
        replace_homophone_char: bool = False,
        delete_random_char: bool = False,
        swap_random_char: bool = False,
        replace_equivalent_num: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialization method. All augmentation methods use default parameters
        in default. We recommend you to only use 1-3 augmentation methods at a
        time. Otherwise, the semantics of samples might be changed
        significantly. **Notice**: some augmentation method might not work for
        some special texts, so there might be no augmented texts generated.

        :param sequential: whether combine all augmentation methods to a
            sequence. If it's True, a sample will be augmented by all opened
            augmentation methods sequentially. If it's False, each opened
            augmentation method would generate its augmented samples
            independently.
        :param aug_num: number of augmented samples to be generated. If
            `sequential` is True, there will be total aug_num augmented samples
            generated. If it's False, there will be (aug_num *
            #opened_aug_method) augmented samples generated.
        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only generated texts in the final
            datasets and the original texts will be removed. It's True in
            default.
        :param replace_similar_word: whether to open the augmentation method of
            replacing random words with their similar words in the original
            texts. e.g. "这里一共有5种不同的数据增强方法" --> "这边一共有5种不同的数据增强方法"
        :param replace_homophone_char: whether to open the augmentation method
            of replacing random characters with their homophones in the
            original texts. e.g. "这里一共有5种不同的数据增强方法" --> "这里一共有5种不同的濖据增强方法"
        :param delete_random_char: whether to open the augmentation method of
            deleting random characters from the original texts. e.g.
            "这里一共有5种不同的数据增强方法" --> "这里一共有5种不同的数据增强"
        :param swap_random_char: whether to open the augmentation method of
            swapping random contiguous characters in the original texts. e.g.
            "这里一共有5种不同的数据增强方法" --> "这里一共有5种不同的数据强增方法"
        :param replace_equivalent_num: whether to open the augmentation method
            of replacing random numbers with their equivalent representations
            in the original texts. **Notice**: Only for numbers for now. e.g.
            "这里一共有5种不同的数据增强方法" --> "这里一共有伍种不同的数据增强方法"
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        self.aug_num = aug_num
        if aug_num >= 10:
            logger.warning(
                f"Relatively large augmentation number [{aug_num}]"
                f" might generate large number of new samples and "
                f"requires more memory and disk space."
            )
        self.sequential = sequential
        self.keep_original_sample = keep_original_sample

        # hide the redundant outputs from nlpcda library
        with HiddenPrints():
            import warnings

            warnings.filterwarnings("ignore")

            self.aug_pipeline = []
            # sample level

            # word level
            if replace_similar_word:
                # the first sample of augmented sample list is the same as the
                # original sample, so we need generate one more augmented
                # sample to get the expected number of augmented samples. Same
                # below
                create_num = (self.aug_num + 1) if not self.sequential or len(self.aug_pipeline) == 0 else 2
                self.aug_pipeline.append(nlpcda.Similarword(create_num=create_num))

            # char level
            if replace_homophone_char:
                create_num = (self.aug_num + 1) if not self.sequential or len(self.aug_pipeline) == 0 else 2
                self.aug_pipeline.append(nlpcda.Homophone(create_num=create_num))
            if delete_random_char:
                create_num = (self.aug_num + 1) if not self.sequential or len(self.aug_pipeline) == 0 else 2
                self.aug_pipeline.append(nlpcda.RandomDeleteChar(create_num=create_num))
            if swap_random_char:
                create_num = (self.aug_num + 1) if not self.sequential or len(self.aug_pipeline) == 0 else 2
                # only use char_gram=1 for relatively minor changes
                self.aug_pipeline.append(nlpcda.CharPositionExchange(create_num=create_num, char_gram=1))

            # only for numbers now
            if replace_equivalent_num:
                create_num = (self.aug_num + 1) if not self.sequential or len(self.aug_pipeline) == 0 else 2
                self.aug_pipeline.append(nlpcda.EquivalentChar(create_num=create_num))

    def process_batched(self, samples):
        # no augmentation methods are opened
        if len(self.aug_pipeline) == 0:
            if self.keep_original_sample:
                return samples
            else:
                return {key: [] for key in samples}

        texts_to_aug = samples[self.text_key]
        res_samples = deepcopy(samples)

        # get augmented texts
        if self.sequential:
            aug_texts = texts_to_aug
            for aug_method in self.aug_pipeline:
                results = []
                for text in aug_texts:
                    # aug and skip the original text
                    result = aug_method.replace(text)
                    results += result[1:] if len(result) > 1 else result
                aug_texts = results[:]
            if len(aug_texts) == 1 and aug_texts[0] == texts_to_aug[0]:
                aug_texts = []
        else:
            # apply each aug method to generate several augmented texts
            aug_texts = []
            for aug_method in self.aug_pipeline:
                aug_texts += aug_method.replace(texts_to_aug[0])[1:]

        # add augmented samples to the batch with other replicate fields
        if self.keep_original_sample:
            res_samples[self.text_key] += aug_texts
        else:
            res_samples[self.text_key] = aug_texts
        # add other replicate fields
        for key in res_samples:
            if key != self.text_key:
                res_samples[key] = res_samples[key] * len(res_samples[self.text_key])
        return res_samples
