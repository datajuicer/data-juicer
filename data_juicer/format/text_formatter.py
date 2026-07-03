import os
from multiprocessing import Pool
from typing import Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger

from data_juicer.utils.cache_utils import DATA_JUICER_CACHE_HOME
from data_juicer.utils.file_utils import find_files_with_suffix
from data_juicer.utils.lazy_loader import LazyLoader

from .formatter import FORMATTERS, LocalFormatter, add_suffixes, unify_format


def _decrypt_and_extract(data_file, fernet_key_bytes, extracted_filetype_path, file_type):
    """
    Worker function for multiprocessing: decrypt *data_file* in-memory and
    then extract its text to *extracted_filetype_path*.

    This function is defined at module level so that it is picklable and can
    be dispatched to a :class:`multiprocessing.Pool`.

    :param data_file: path to the encrypted source file.
    :param fernet_key_bytes: the raw Fernet key as :class:`bytes` (obtained
        from ``fernet._signing_key`` + ``fernet._encryption_key`` is NOT
        used here; instead we pass the original base-64 key bytes so that the
        child process can reconstruct the :class:`~cryptography.fernet.Fernet`
        instance independently).
    :param extracted_filetype_path: directory where the extracted .txt file
        will be written.
    :param file_type: ``".docx"`` or ``".pdf"``.
    """
    from cryptography.fernet import Fernet

    fernet = Fernet(fernet_key_bytes)

    import io

    with open(data_file, "rb") as fh:
        encrypted_data = fh.read()
    plaintext = fernet.decrypt(encrypted_data)
    buf = io.BytesIO(plaintext)
    buf.name = data_file

    if file_type == ".docx":
        extract_txt_from_docx(buf, extracted_filetype_path)
    else:
        extract_txt_from_pdf(buf, extracted_filetype_path)


def extract_txt_from_docx(fn, tgt_path):
    """
    Extract text from a docx file and save to target path.

    :param fn: path to input docx file, or a file-like object (e.g.
        :class:`io.BytesIO`) for in-memory decrypted content.
    :param tgt_path: path to save text file.
    """
    docx = LazyLoader("docx", "python-docx")
    doc = docx.Document(fn)
    text = [para.text for para in doc.paragraphs if para.text.strip()]
    # Derive the output filename from fn; fall back to a generic name for
    # file-like objects that expose a .name attribute.
    if hasattr(fn, "name"):
        base_fn = os.path.basename(fn.name).lower().replace(".docx", ".txt")
    else:
        base_fn = os.path.basename(fn).lower().replace(".docx", ".txt")
    with open(os.path.join(tgt_path, base_fn), "w") as f:
        f.write("\n".join(text))


def extract_txt_from_pdf(fn, tgt_path):
    """
    Extract text from a pdf file and save to target path.

    :param fn: path to input pdf file, or a file-like object (e.g.
        :class:`io.BytesIO`) for in-memory decrypted content.
    :param tgt_path: path to save text file.
    """
    pdfplumber = LazyLoader("pdfplumber")
    with pdfplumber.open(fn) as pdf:
        text = []
        for page in pdf.pages:
            # remove tables from each page extracted by pdfplumber
            tables = page.find_tables()
            for table in tables:
                page = page.outside_bbox(table.bbox)
            # remove page number from the end of each page
            page_text = page.extract_text()
            page_num = str(page.page_number)
            if page_text.rstrip().endswith(page_num):
                page_text = page_text.rstrip()[: -len(page_num)]
            if page_text.strip():
                text.append(page_text)
        if hasattr(fn, "name"):
            base_fn = os.path.basename(fn.name).lower().replace(".pdf", ".txt")
        else:
            base_fn = os.path.basename(fn).lower().replace(".pdf", ".txt")
        with open(os.path.join(tgt_path, base_fn), "w") as f:
            f.write("\n".join(text))


@FORMATTERS.register_module()
class TextFormatter(LocalFormatter):
    """
    The class is used to load and format text-type files.

    e.g. `['.txt', '.pdf', '.cpp', '.docx']`
    """

    SUFFIXES = [
        ".docx",
        ".pdf",
        ".txt",
        ".md",
        ".tex",
        ".asm",
        ".bat",
        ".cmd",
        ".c",
        ".h",
        ".cs",
        ".cpp",
        ".hpp",
        ".c++",
        ".h++",
        ".cc",
        ".hh",
        ".C",
        ".H",
        ".cmake",
        ".css",
        ".dockerfile",
        ".f90",
        ".f",
        ".f03",
        ".f08",
        ".f77",
        ".f95",
        ".for",
        ".fpp",
        ".go",
        ".hs",
        ".html",
        ".java",
        ".js",
        ".jl",
        ".lua",
        ".markdown",
        ".php",
        ".php3",
        ".php4",
        ".php5",
        ".phps",
        ".phpt",
        ".pl",
        ".pm",
        ".pod",
        ".perl",
        ".ps1",
        ".psd1",
        ".psm1",
        ".py",
        ".rb",
        ".rs",
        ".sql",
        ".scala",
        ".sh",
        ".bash",
        ".command",
        ".zsh",
        ".ts",
        ".tsx",
        ".vb",
        "Dockerfile",
        "Makefile",
        ".xml",
        ".rst",
        ".m",
        ".smali",
    ]

    def __init__(self, dataset_path, suffixes=None, add_suffix=False, **kwargs):
        """
        Initialization method.

        :param dataset_path: a dataset file or a dataset directory
        :param suffixes: files with specified suffixes to be processed
        :param add_suffix: Whether to add file suffix to dataset meta
            info
        :param kwargs: extra args
        """
        super().__init__(
            dataset_path=dataset_path,
            suffixes=suffixes if suffixes else self.SUFFIXES,
            type="text",
            add_suffix=add_suffix,
            **kwargs,
        )
        self.dataset_path = dataset_path
        self.add_suffix = add_suffix

    def load_dataset(self, num_proc: Optional[int] = None, global_cfg=None) -> Dataset:
        """
        Load a dataset from local text-type files.

        :param num_proc: number of processes when loading the dataset
        :param global_cfg: the global cfg used in consequent processes,
        :return: unified_format_dataset.
        """
        # extract text to cache directory
        extracted_dataset_path = os.path.join(
            DATA_JUICER_CACHE_HOME, os.path.basename(os.path.abspath(self.dataset_path))
        )

        # Determine whether to decrypt files in-memory before extraction.
        decrypt = global_cfg is not None and getattr(global_cfg, "decrypt_after_reading", False)
        fernet = None
        if decrypt:
            from data_juicer.utils.encryption_utils import load_fernet_key

            fernet = load_fernet_key(getattr(global_cfg, "encryption_key_path", None))

        # Resolve num_proc early so Pool() calls below use the correct value.
        _num_proc = self.kwargs.pop("num_proc", 1)
        num_proc = num_proc or _num_proc

        for file_type in self.data_files:
            # extract text from docx or pdf files, and save as txt type
            if file_type == ".docx" or file_type == ".pdf":
                extracted_filetype_path = os.path.join(extracted_dataset_path, file_type.strip("."))
                if not os.path.exists(extracted_filetype_path):
                    os.makedirs(extracted_filetype_path)
                logger.info("Extracting text from {} files...".format(file_type.strip(".")))

                extract_func = extract_txt_from_docx if file_type == ".docx" else extract_txt_from_pdf

                if decrypt and fernet is not None:
                    # Decrypt each file in-memory and extract text in parallel.
                    # BytesIO objects are not picklable, so we pass the raw
                    # Fernet key bytes to each worker; the worker reconstructs
                    # its own Fernet instance and decrypts independently.
                    fernet_key_bytes = fernet._signing_key + fernet._encryption_key
                    pool = Pool(num_proc)
                    for data_file in self.data_files[file_type]:
                        pool.apply_async(
                            func=_decrypt_and_extract,
                            args=(
                                data_file,
                                fernet_key_bytes,
                                extracted_filetype_path,
                                file_type,
                            ),
                        )
                    pool.close()
                    pool.join()
                else:
                    pool = Pool(num_proc)
                    for data_file in self.data_files[file_type]:
                        pool.apply_async(
                            func=extract_func,
                            args=(
                                data_file,
                                extracted_filetype_path,
                            ),
                        )
                    pool.close()
                    pool.join()
                logger.info(f"Extracted text files are stored in directory " f"{extracted_filetype_path}")

                # look for extracted txt files
                self.data_files[file_type] = find_files_with_suffix(extracted_filetype_path, ".txt")[".txt"]

        # load text dataset, one text file as one sample
        logger.info(f"Loading dataset with num_proc: {num_proc}")

        # For plain text files (not PDF/DOCX), apply in-memory decryption via
        # temporary files: decrypt to a NamedTemporaryFile, load, then delete.
        if decrypt and fernet is not None:
            import tempfile

            from data_juicer.utils.encryption_utils import (
                decrypt_file_to_bytes,
                get_secure_tmpdir,
            )

            decrypted_data_files = {}
            tmp_paths = []
            tmp_dir = get_secure_tmpdir()
            for ext, paths in self.data_files.items():
                # PDF/DOCX files have already been extracted to .txt above;
                # the remaining types need decryption here.
                tmp_file_paths = []
                for p in paths:
                    data = decrypt_file_to_bytes(p, fernet)
                    suffix = os.path.splitext(p)[-1]
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=tmp_dir) as tmp:
                        tmp.write(data)
                        tmp_paths.append(tmp.name)
                        tmp_file_paths.append(tmp.name)
                decrypted_data_files[ext.strip(".")] = tmp_file_paths
            try:
                datasets = load_dataset(
                    "text",
                    data_files=decrypted_data_files,
                    sample_by="document",
                    num_proc=num_proc,
                    **self.kwargs,
                )
            finally:
                for tp in tmp_paths:
                    try:
                        os.remove(tp)
                    except OSError:
                        pass
        else:
            datasets = load_dataset(
                "text",
                data_files={key.strip("."): self.data_files[key] for key in self.data_files},
                sample_by="document",
                num_proc=num_proc,
                **self.kwargs,
            )
        # whether to add file suffix to dataset meta info
        if self.add_suffix:
            logger.info("Add suffix info into dataset...")
            datasets = add_suffixes(datasets, num_proc)
        else:
            datasets = concatenate_datasets([ds for _, ds in datasets.items()])
        return unify_format(datasets, text_keys=self.text_keys, num_proc=num_proc, global_cfg=global_cfg)
