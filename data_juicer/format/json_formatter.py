import os

from loguru import logger

from data_juicer.format.formatter import FORMATTERS, LocalFormatter, unify_format
from data_juicer.utils.jsonl_lenient_loader import (
    JSONL_LENIENT_EXTENSIONS,
    dataset_from_lenient_jsonl_files,
)


@FORMATTERS.register_module()
class JsonFormatter(LocalFormatter):
    """
    Load json-type files.

    Default suffixes include ``.json``, ``.jsonl``, gzip/zstd variants.

    Optional lenient JSONL: ``load_jsonl_lenient: true`` or env
    ``DATA_JUICER_JSONL_LENIENT=1`` streams jsonl-only inputs with stdlib
    :func:`json.loads`, skipping bad lines (avoids HF ujson for those files).
    """

    SUFFIXES = [
        ".json",
        ".jsonl",
        ".json.gz",
        ".jsonl.gz",
        ".json.zst",
        ".jsonl.zst",
    ]

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
            type="json",
            **kwargs,
        )

    def load_dataset(self, num_proc=None, global_cfg=None):
        env_key = "DATA_JUICER_JSONL_LENIENT"
        env_raw = os.environ.get(env_key, "").strip().lower()
        lenient_env = env_raw in ("1", "true", "yes", "on")
        lenient_cfg = False
        if global_cfg is not None:
            lenient_cfg = bool(
                getattr(global_cfg, "load_jsonl_lenient", False),
            )
        lenient = lenient_cfg or lenient_env

        if not lenient:
            return super().load_dataset(num_proc, global_cfg)

        # Stream jsonl shards only; skipping .json avoids HF/ujson "too big".
        jsonl_only_files = {}
        skipped_by_ext = {}
        for ext, paths in self.data_files.items():
            if ext in JSONL_LENIENT_EXTENSIONS:
                jsonl_only_files[ext] = paths
            else:
                skipped_by_ext[ext] = paths
        if skipped_by_ext:
            for ext, paths in sorted(skipped_by_ext.items()):
                n = len(paths)
                logger.warning(
                    f"[lenient jsonl] Ignoring {n} file(s) {ext!r}; " f"lenient mode only reads jsonl* shards."
                )

        file_ext_pairs = []
        for ext, files in sorted(jsonl_only_files.items()):
            for fp in sorted(files):
                file_ext_pairs.append((fp, ext))

        if not file_ext_pairs:
            msg = "[lenient jsonl] No .jsonl/.jsonl.gz/.jsonl.zst matched; " "using default HuggingFace JSON loader."
            logger.warning(msg)
            return super().load_dataset(num_proc, global_cfg)

        nsh = len(file_ext_pairs)
        logger.info(f"[lenient jsonl] ACTIVE: streaming {nsh} jsonl shard(s) " f"with stdlib json (bad lines skipped).")

        _num_proc = self.kwargs.pop("num_proc", 1)
        num_proc_eff = num_proc or _num_proc
        if num_proc_eff != 1:
            logger.info("[lenient jsonl] Single-threaded load; ignoring num_proc>1.")

        ds = dataset_from_lenient_jsonl_files(
            file_ext_pairs,
            add_suffix_column=self.add_suffix,
        )
        return unify_format(
            ds,
            text_keys=self.text_keys,
            num_proc=num_proc_eff,
            global_cfg=global_cfg,
        )
