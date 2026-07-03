import tarfile
import zipfile

from loguru import logger

from ..base_op import OPERATORS, Mapper

OP_NAME = "latex_merge_tex_mapper"


@OPERATORS.register_module(OP_NAME)
class LatexMergeTexMapper(Mapper):
    """Extracts and concatenates all ``.tex`` files from a compressed
    LaTeX project archive into a single text field.

    Supported archive formats: ``.tar``, ``.tar.gz`` / ``.tgz``,
    and ``.zip``.  Plain ``.gz`` (single-file gzip) is **not**
    supported because gzip archives carry no filename metadata,
    making it impossible to verify that the content is actually a
    ``.tex`` file.

    All ``.tex`` files found inside the archive are read in-memory and
    joined with a configurable separator.  No ordering or
    deduplication is applied.

    This operator is typically placed before LaTeX-processing operators
    such as ``remove_comments_mapper``, ``expand_macro_mapper``, or
    ``latex_figure_context_extractor_mapper``."""

    def __init__(
        self,
        compressed_file_key: str = "compressed_file",
        separator: str = "\n\n",
        max_file_size: int = 50 * 1024 * 1024,
        max_total_size: int = 100 * 1024 * 1024,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param compressed_file_key: Field name that stores the archive
            file path.
        :param separator: String used to join the contents of multiple
            ``.tex`` files.
        :param max_file_size: Maximum allowed uncompressed size in bytes
            for a single ``.tex`` entry inside the archive.  Entries
            exceeding this limit are skipped with a warning.  Set to
            ``None`` or ``0`` to disable the check.
        :param max_total_size: Maximum allowed cumulative size in bytes
            for all extracted ``.tex`` content combined.  Once this
            limit is reached, remaining files in the archive are
            skipped with a warning.  Set to ``None`` or ``0`` to
            disable the check.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.compressed_file_key = compressed_file_key
        self.separator = separator
        self.max_file_size = max_file_size or 0
        self.max_total_size = max_total_size or 0

    def _extract_tex_contents(self, archive_path: str):
        """Return a list of decoded ``.tex`` file contents from
        *archive_path*.  Dispatches by file extension to the
        appropriate reader."""
        path_lower = archive_path.lower()

        try:
            if path_lower.endswith(".zip"):
                return self._read_zip(archive_path, self.max_file_size, self.max_total_size)
            elif path_lower.endswith((".tar.gz", ".tgz", ".tar")):
                return self._read_tar(archive_path, self.max_file_size, self.max_total_size)
            else:
                logger.warning(
                    f"Unsupported archive format: {archive_path}. " f"Supported formats: .tar, .tar.gz, .tgz, .zip"
                )
                return []
        except Exception:
            logger.exception(f"Failed to read archive {archive_path}")
            return []

    @staticmethod
    def _read_tar(archive_path: str, max_file_size: int = 0, max_total_size: int = 0):
        contents = []
        total_bytes = 0
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                if not member.name.endswith(".tex"):
                    continue
                if max_file_size and member.size > max_file_size:
                    logger.warning(
                        f"Skipping {member.name} in {archive_path}: "
                        f"declared size {member.size} bytes exceeds "
                        f"limit of {max_file_size} bytes"
                    )
                    continue
                # Use declared header size to bail before reading.
                if max_total_size and (total_bytes + member.size) > max_total_size:
                    logger.warning(
                        f"Cumulative extracted size would exceed limit "
                        f"of {max_total_size} bytes in {archive_path}. "
                        f"Skipping remaining files."
                    )
                    break
                raw = tf.extractfile(member)
                if raw is None:
                    continue
                raw_bytes = raw.read()
                if max_file_size and len(raw_bytes) > max_file_size:
                    logger.warning(
                        f"Skipping {member.name} in {archive_path}: "
                        f"actual size {len(raw_bytes)} bytes exceeds "
                        f"limit of {max_file_size} bytes"
                    )
                    continue
                total_bytes += len(raw_bytes)
                if max_total_size and total_bytes > max_total_size:
                    logger.warning(
                        f"Cumulative extracted size {total_bytes} bytes "
                        f"exceeds limit of {max_total_size} bytes in "
                        f"{archive_path}. Skipping remaining files."
                    )
                    break
                contents.append(raw_bytes.decode("utf-8", errors="replace"))
        return contents

    @staticmethod
    def _read_zip(archive_path: str, max_file_size: int = 0, max_total_size: int = 0):
        contents = []
        total_bytes = 0
        with zipfile.ZipFile(archive_path) as zf:
            for name in zf.namelist():
                if not name.endswith(".tex"):
                    continue
                info = zf.getinfo(name)
                if max_file_size and info.file_size > max_file_size:
                    logger.warning(
                        f"Skipping {name} in {archive_path}: "
                        f"declared size {info.file_size} bytes exceeds "
                        f"limit of {max_file_size} bytes"
                    )
                    continue
                # Use declared header size to bail before reading.
                if max_total_size and (total_bytes + info.file_size) > max_total_size:
                    logger.warning(
                        f"Cumulative extracted size would exceed limit "
                        f"of {max_total_size} bytes in {archive_path}. "
                        f"Skipping remaining files."
                    )
                    break
                raw_bytes = zf.read(name)
                if max_file_size and len(raw_bytes) > max_file_size:
                    logger.warning(
                        f"Skipping {name} in {archive_path}: "
                        f"actual size {len(raw_bytes)} bytes exceeds "
                        f"limit of {max_file_size} bytes"
                    )
                    continue
                total_bytes += len(raw_bytes)
                if max_total_size and total_bytes > max_total_size:
                    logger.warning(
                        f"Cumulative extracted size {total_bytes} bytes "
                        f"exceeds limit of {max_total_size} bytes in "
                        f"{archive_path}. Skipping remaining files."
                    )
                    break
                contents.append(raw_bytes.decode("utf-8", errors="replace"))
        return contents

    def process_single(self, sample):
        if self.compressed_file_key not in sample:
            raise ValueError(
                f"Compressed file key '{self.compressed_file_key}' "
                f"not found in sample. "
                f"Available keys: {list(sample.keys())}"
            )

        path = sample[self.compressed_file_key]
        tex_contents = self._extract_tex_contents(path)
        sample[self.text_key] = self.separator.join(tex_contents)
        return sample
