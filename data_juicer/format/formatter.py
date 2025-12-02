import os
from typing import List, Union

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from jsonargparse import Namespace, dict_to_namespace
from loguru import logger

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import find_files_with_suffix, is_absolute_path
from data_juicer.utils.registry import Registry

FORMATTERS = Registry("Formatters")


class BaseFormatter:
    """Base class to load dataset."""

    def load_dataset(self, *args) -> Dataset:
        raise NotImplementedError


class LocalFormatter(BaseFormatter):
    """The class is used to load a dataset from local files or local
    directory."""

    def __init__(
        self,
        dataset_path: str,
        type: str,
        suffixes: Union[str, List[str], None] = None,
        text_keys: List[str] = None,
        add_suffix=False,
        **kwargs,
    ):
        """
        Initialization method.

        :param dataset_path: path to a dataset file or a dataset
            directory
        :param type: a packaged dataset module type (json, csv, etc.)
        :param suffixes: files with specified suffixes to be processed
        :param text_keys: key names of field that stores sample
            text.
        :param add_suffix: whether to add the file suffix to dataset
            meta info
        :param kwargs: extra args
        """
        self.type = type
        self.kwargs = kwargs
        self.text_keys = text_keys
        self.data_files = find_files_with_suffix(dataset_path, suffixes)
        self.add_suffix = add_suffix

    def load_dataset(self, num_proc: int = 1, global_cfg=None) -> Dataset:
        """
        Load a dataset from dataset file or dataset directory, and unify its
        format.

        :param num_proc: number of processes when loading the dataset
        :param global_cfg: global cfg used in consequent processes,
        :return: formatted dataset
        """
        _num_proc = self.kwargs.pop("num_proc", 1)
        num_proc = num_proc or _num_proc
        datasets = load_dataset(
            self.type,
            data_files={key.strip("."): self.data_files[key] for key in self.data_files},
            num_proc=num_proc,
            **self.kwargs,
        )
        if self.add_suffix:
            logger.info("Add suffix info into dataset...")
            datasets = add_suffixes(datasets, num_proc)
        else:
            from data_juicer.core.data import NestedDataset

            datasets = NestedDataset(concatenate_datasets([ds for _, ds in datasets.items()]))
        ds = unify_format(datasets, text_keys=self.text_keys, num_proc=num_proc, global_cfg=global_cfg)
        return ds


class RemoteFormatter(BaseFormatter):
    """The class is used to load a dataset from repository of huggingface
    hub."""

    def __init__(self, dataset_path: str, text_keys: List[str] = None, **kwargs):
        """
        Initialization method.

        :param dataset_path: a dataset file or a dataset directory
        :param text_keys: key names of field that stores sample
            text.
        :param kwargs: extra args
        """
        self.path = dataset_path
        self.text_keys = text_keys
        self.kwargs = kwargs

    def load_dataset(self, num_proc: int = 1, global_cfg=None) -> Dataset:
        """
        Load a dataset from HuggingFace, and unify its format.

        :param num_proc: number of processes when loading the dataset
        :param global_cfg: the global cfg used in consequent processes,
        :return: formatted dataset
        """
        ds = load_dataset(self.path, split="train", num_proc=num_proc, **self.kwargs)
        ds = unify_format(ds, text_keys=self.text_keys, num_proc=num_proc, global_cfg=global_cfg)
        return ds


def add_suffixes(datasets: DatasetDict, num_proc: int = 1) -> Dataset:
    """
    Add suffix filed to datasets.

    :param datasets: a DatasetDict object
    :param num_proc: number of processes to add suffixes
    :return: datasets with suffix features.
    """
    logger.info("Add suffix column for dataset")
    from data_juicer.core.data import add_same_content_to_new_column

    for key, ds in datasets.items():
        if Fields.suffix not in ds.features:
            datasets[key] = ds.map(
                add_same_content_to_new_column,
                fn_kwargs={"new_column_name": Fields.suffix, "initial_value": "." + key},
                num_proc=num_proc,
                desc="Adding new column for suffix",
            )
    datasets = concatenate_datasets([ds for _, ds in datasets.items()])
    from data_juicer.core.data import NestedDataset

    return NestedDataset(datasets)


def unify_format(
    dataset: Dataset,
    text_keys: Union[List[str], str] = "text",
    num_proc: int = 1,
    global_cfg: Union[dict, Namespace] = None,
) -> Dataset:
    """
    Get an unified internal format, conduct the following modifications.

    1. check keys of dataset

    2. filter out those samples with empty or None text

    :param dataset: input dataset
    :param text_keys: original text key(s) of dataset.
    :param num_proc: number of processes for mapping
    :param global_cfg: the global cfg used in consequent processes,
        since cfg.text_key may be modified after unifying

    :return: unified_format_dataset
    """
    from data_juicer.core.data import NestedDataset

    if isinstance(dataset, DatasetDict):
        datasets = list(dataset.values())
        assert len(datasets) == 1, "Please make sure the passed datasets " "contains only 1 dataset"
        dataset = datasets[0]
    assert isinstance(dataset, Dataset) or isinstance(dataset, NestedDataset), (
        "Currently we only support processing data" "with huggingface-Dataset format"
    )

    if text_keys is None:
        text_keys = []

    if isinstance(text_keys, str):
        text_keys = [text_keys]

    logger.info("Unifying the input dataset formats...")

    dataset = NestedDataset(dataset)

    # 1. check text related keys
    for key in text_keys:
        if key not in dataset.features:
            err_msg = (
                f"There is no key [{key}] in dataset. You might set "
                f"wrong text_key in the config file for your dataset. "
                f"Please check and retry!"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    # 2. filter out those samples with empty or None text
    # TODO: optimize the filtering operation for better efficiency
    logger.info(f"There are {len(dataset)} sample(s) in the original dataset.")

    def non_empty_text(sample, target_keys):
        for target_key in target_keys:
            # TODO: case for CFT, in which the len(sample[target_key]) == 0
            if sample[target_key] is None:
                # we filter out the samples contains at least None column
                # since the op can not handle it now
                return False
        return True

    dataset = dataset.filter(non_empty_text, num_proc=num_proc, fn_kwargs={"target_keys": text_keys})
    logger.info(f"{len(dataset)} samples left after filtering empty text.")

    # 3. convert relative paths to absolute paths
    if global_cfg is not None:
        if isinstance(global_cfg, dict):
            global_cfg = dict_to_namespace(global_cfg)
        # check and get dataset dir
        if (
            hasattr(global_cfg, "dataset_path")
            and global_cfg.dataset_path is not None
            and os.path.exists(global_cfg.dataset_path)
        ):
            if os.path.isdir(global_cfg.dataset_path):
                ds_dir = global_cfg.dataset_path
            else:
                ds_dir = os.path.dirname(global_cfg.dataset_path)
        else:
            ds_dir = ""
        image_key = global_cfg.image_key if hasattr(global_cfg, "image_key") else "images"
        audio_key = global_cfg.audio_key if hasattr(global_cfg, "audio_key") else "audios"
        video_key = global_cfg.video_key if hasattr(global_cfg, "video_key") else "videos"

        data_path_keys = []
        if image_key in dataset.features:
            data_path_keys.append(image_key)
        if audio_key in dataset.features:
            data_path_keys.append(audio_key)
        if video_key in dataset.features:
            data_path_keys.append(video_key)
        if len(data_path_keys) == 0:
            # no image/audio/video path list in dataset, no need to convert
            return dataset

        if ds_dir == "":
            return dataset

        logger.info(
            "Converting relative paths in the dataset to their "
            "absolute version. (Based on the directory of input "
            "dataset file)"
        )

        # function to convert relative paths to absolute paths
        def rel2abs(sample, path_keys, dataset_dir):
            for path_key in path_keys:
                if path_key not in sample:
                    continue
                paths = sample[path_key]
                if not paths:
                    continue
                new_paths = [path if is_absolute_path(path) else os.path.join(dataset_dir, path) for path in paths]
                sample[path_key] = new_paths
            return sample

        dataset = dataset.map(
            rel2abs, num_proc=num_proc, fn_kwargs={"path_keys": data_path_keys, "dataset_dir": ds_dir}
        )
    else:
        logger.warning(
            "No global config passed into unify_format function. "
            "Relative paths in the dataset might not be converted "
            "to their absolute versions. Data of other modalities "
            "might not be able to find by Data-Juicer."
        )

    return dataset
