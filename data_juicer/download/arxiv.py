import gzip
import os
import re
import subprocess
import tarfile
import tempfile

from datasets import Dataset
from downloader import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
    download_and_extract,
    get_arxiv_urls,
)

from data_juicer.utils.file_utils import (
    expand_outdir_and_mkdir,
    get_all_files_paths_under,
)

# The iterator and extractor code are in large part taken
# from the Red-Pajama repo
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/arxiv


class ArxivDownloader(DocumentDownloader):
    def __init__(self, download_dir, verbose=False):
        super().__init__()
        self._download_dir = download_dir
        self._verbose = False

    def download(self, tarfile):
        output_file = os.path.join(self._download_dir, tarfile)
        s3path = os.path.join("s3://arxiv/src", tarfile)
        if os.path.exists(output_file):
            print(f"tar file: {output_file} exists. Not downloading")
        else:
            print(f"Downloading {s3path} and writing to {output_file}")
            cmd = ["s5cmd", "--request-payer=requester", "cp", s3path, output_file]
            if self._verbose:
                stdout, stderr = None, None
            else:
                stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
            p = subprocess.run(
                cmd,
                stdout=stdout,
                stderr=stderr,
            )
            if p.returncode != 0:
                print(f"Failed to download {s3path} to {output_file}")

        return output_file


class ArxivIterator(DocumentIterator):
    def __init__(self, log_frequency=1000):
        super().__init__()
        self._log_freq = log_frequency
        self._cnt = 0

    def iterate(self, file_path):
        self._cnt = 0
        download_dir = os.path.split(file_path)[0]
        bname = os.path.split(file_path)[-1]
        # with (tempfile...), yapf spits lib2to3.pgen2.parse.ParseError
        with tempfile.TemporaryDirectory(dir=download_dir) as tmpdir:
            with tarfile.open(file_path) as tf:
                tf.extractall(members=tf.getmembers(), path=tmpdir)
                for i, item in enumerate(get_all_files_paths_under(tmpdir)):
                    if self._cnt > 0 and self._cnt % self._log_freq == 0:
                        print(f"Extracted {self._cnt} papers from {file_path}")
                    self._cnt += 1

                    tex_files = self._tex_proj_loader(item)
                    arxiv_id = os.path.splitext(os.path.split(item)[-1])[0]

                    # get the arxiv id in the correct format
                    try:
                        clean_arxiv_id = self._format_arxiv_id(arxiv_id)
                    except Exception as e:
                        print(f"[WARNING] failed to format arxiv id {arxiv_id}; exception={e}")  # noqa: E501
                        clean_arxiv_id = arxiv_id

                    if tex_files is None:
                        continue

                    yield {"id": clean_arxiv_id, "source_id": f"{bname}"}, tex_files

    def _tex_proj_loader(self, file_or_dir_path):
        r"""function to load the tex files from a tar file or a gzip file. The
        function will return a tuple containing a list of tex files and the
        timestamp of the project.

        @param file_or_dir_path: path to the tar file or the gzip file

        @return: tuple containing a list of tex files and the timestamp of the
            project
        """  # noqa E501
        files_and_content = []

        try:
            # if it is a directory, open it as a tarfile
            with tarfile.open(file_or_dir_path) as sub_tf:
                for member in sub_tf.getmembers():
                    if member.name.endswith(".tex"):
                        file_content = sub_tf.extractfile(member).read()

                        try:
                            file_content = file_content.decode("utf-8")
                        except UnicodeDecodeError:
                            return None

                        files_and_content.append(file_content)

        except tarfile.ReadError:
            # otherwise we try opening it as a gzip file
            try:
                with gzip.open(file_or_dir_path, "rb") as gz:
                    file_content = gz.read()
            except Exception:
                # all fails, we skip this file
                # self._logger.info(f"[ERROR] {e}: {file_or_dir_path}")
                return None

            try:
                file_content = file_content.decode("utf-8")
            except UnicodeDecodeError:
                # self._logger.info(f"UnicodeDecodeError: {file_or_dir_path}")
                return None

            files_and_content.append(file_content)

        except Exception as e:
            print(f"[ERROR] {e}: {file_or_dir_path}")
            return None

        return files_and_content

    def _format_arxiv_id(self, arxiv_id):
        r"""this function brings the raw arxiv-id into a format compliant with the
        specification from arxiv. This is used to create the url to the arxiv
        abstract page.

        - Format prior to March 2007:
            <archive>/YYMMNNN where N is a 3-digit number
        - Format after March 2007: <archive>/YYMM.NNNNN where N is a
          5 (or 6)-digit number

        References: https://info.arxiv.org/help/arxiv_identifier.html

        @param arxiv_id: raw arxiv id which can be in one of the
                         following formats:
                         - <archive><YY><MM><NNN>
                         - <YY><MM><NNNNN|NNNNNN>

        @return: formatted arxiv id
        """  # noqa: E501
        match = re.search(r"^([a-zA-Z-]*)([\d\.]+)$", arxiv_id)

        if match is None:
            raise ValueError(f"Invalid arxiv id: {arxiv_id}")

        if match.group(1) == "":
            return match.group(2)

        return f"{match.group(1)}/{match.group(2)}"


class ArxivExtractor(DocumentExtractor):
    def __init__(self):
        super().__init__()

    def extract(self, content):
        if len(content) == 0:
            return None

        # build dictionaries that contain the definitions of all
        # macros in all text files. This is later used to expand
        # all macros used in the text with their definitions, so
        # that consistency among different authors is ensured

        non_arg_macros = {}
        for file_content in content:
            non_arg_macros.update(self._build_non_arg_macros_dict(file_content))

        # TODO: macros that take arguments are not supported yet
        arg_macros = {}

        # join multiple latex files with a newline character
        try:
            cleaned_latex_file_str = "\n".join(
                self._clean_tex_file(
                    file_content=file_content,
                    arg_macros=arg_macros,
                    non_arg_macros=non_arg_macros,
                )
                for file_content in content
            )
        except Exception:
            return {}, None

        # Don't return meta
        if cleaned_latex_file_str is not None:
            if len(cleaned_latex_file_str) > 0:
                return {}, cleaned_latex_file_str

    def _clean_tex_file(self, file_content, arg_macros, non_arg_macros):
        r"""function takes a tex file as input and returns a cleaned version. The
         cleaned version is a concatenation of the tex files with the
        following modifications:

        - remove all comments (i.e. all lines starting with %)
        - remove everything before the first section-like header
        - remove everything after the first occurrence of either \appendix or
            \bibliography
        - inline-expand definitions and macros

        @param file_content: the content of the tex file as a string.

        @return: cleaned tex file as a string
        """  # noqa: E501
        # find the first occurrence of a \section-like header and replace
        # everything before it with an empty string. This matches the
        # following pattern: \<section-type>[optional-args]{name}
        pattern = r"^(.*?)("
        pattern += r"\\\bchapter\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bpart\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsubsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
        pattern += r"\\\bsubparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
        pattern += r")"

        # if no section like header is found, then we return an empty string
        if not re.search(pattern, file_content, flags=re.DOTALL):
            return ""

        # replace everything with the second group of the match
        # (i.e. everything after and including the section header)
        file_content = re.sub(
            pattern=pattern,
            repl=r"\2",
            string=file_content,
            flags=re.DOTALL,  # make sure that the dot matches also newlines
        )

        # remove all line comments
        file_content = re.sub(
            pattern=r"(?m)^%.*\n?",
            repl=r"",
            string=file_content,
            flags=re.MULTILINE,
        )

        # remove all in comments within a line
        file_content = re.sub(
            # pattern matches a "%" that is not preceded by a backslash
            pattern=r"[^\\]%.+$",
            repl=r"",
            string=file_content,
            flags=re.MULTILINE,
        )

        # find the first occurrence of either \appendix or \bibliography and
        # replace everything after it with an empty string
        pattern = r"("
        pattern += r"\\appendix|"
        pattern += r"\\begin\{references\}|"
        pattern += r"\\begin\{REFERENCES\}|"
        pattern += r"\\begin\{thebibliography\}|"
        pattern += r"\\bibliography\{.*\}"
        pattern += r").*$"

        file_content = re.sub(
            pattern=pattern,
            repl=r"",
            string=file_content,
            flags=re.DOTALL,  # make sure that the dot matches also newlines
        )

        # inline-expand all non-arg macros
        for macro_name, macro_value in non_arg_macros.items():
            file_content = re.sub(
                # make pattern grouped to make sure that the macro is not part
                # of a longer alphanumeric word
                pattern=r"(" + macro_name + r")" + r"([^a-zA-Z0-9])",
                # replace the macro with its value and add back the character
                # that was matched after the macro
                repl=macro_value + r"\2",
                string=file_content,
            )

        # inline-expand all macros that use args
        # TODO: inline-expand macros with args
        for macro_name, macro_value in arg_macros.items():
            pass

        return file_content

    def _build_non_arg_macros_dict(self, file_content):
        r"""function takes the content of a tex file and returns a dictionary
        that contains the definitions of all macros that do not use arguments.
        The dictionary is of the form {macro_name: macro_value}.

        @param file_content: the content of the tex file as a string.

        @return: dict
        """
        # regex for extracting \newcommand macros without arguments
        non_arg_nc_reg = re.compile(
            # this regex matches the following:
            # \newcommand{\macro_name}{macro_value}
            # \newcommand*{\macro_name}{macro_value}
            # where macro_name is only allowed to contain letters and numbers;
            # macro_value can contain any character.
            pattern=r"\\\bnewcommand\b\*?\{(\\[a-zA-Z0-9]+?)\}\{(.*?)\}$",
            flags=re.MULTILINE,
        )

        # regex for extracting \def macros without arguments
        non_arg_def_reg = re.compile(
            # this regex matches the following:
            # \def\macro_name{macro_value}
            # where macro_name is only allowed to contain letters and numbers;
            # macro_value can contain any character.
            pattern=r"\\def\s*(\\[a-zA-Z0-9]+?)\s*\{(.*?)\}$",
            flags=re.MULTILINE,
        )

        # Extract all user-defined LaTeX macros from the preamble
        macros = {}
        for reg in [non_arg_nc_reg, non_arg_def_reg]:
            for match in reg.finditer(file_content):
                # convert the macro name and value to a raw string that can be
                # used in re.sub
                macro_name = match.group(1).encode("unicode-escape").decode("utf-8")
                macro_val = match.group(2).encode("unicode-escape").decode("utf-8")

                macros[macro_name] = macro_val

        return macros


def download_arxiv(
    output_path: str,
    output_type: str = "jsonl",
    raw_download_dir=None,
    keep_raw_download=False,
    force_download=False,
    url_limit=None,
) -> Dataset:
    """
    Downloads Arxiv tar files and extracts them

    Args:
      output_path: The path to the root directory of the files
      output_type: The file type to save the data as.
      raw_download_dir: Path to store the raw download files for intermediate processing.
        If None, they are stored in a folder named "downloads" under output_path.
      keep_raw_download: If True, keeps the compressed WARC files that have not been extracted.
      force_download: If False, will skip processing all files in output_paths that already exist and
        directly read from them instead.
      url_limit: The maximum number of raw files to download from the snapshot. If None, all
        files from the range of snapshots are downloaded.
    """  # noqa: E501
    arxiv_urls = get_arxiv_urls()
    if url_limit:
        arxiv_urls = arxiv_urls[:url_limit]
    output_paths = list(map(lambda url: os.path.join(output_path, f"{url}.{output_type}"), arxiv_urls))

    if not raw_download_dir:
        raw_download_dir = os.path.join(output_path, "downloads")
    expand_outdir_and_mkdir(raw_download_dir)
    downloader = ArxivDownloader(raw_download_dir)
    iterator = ArxivIterator()
    extractor = ArxivExtractor()

    output_format = {
        "text": str,
        "id": str,
        "source_id": str,
        "filename": str,
    }
    dataset = download_and_extract(
        arxiv_urls,
        output_paths,
        downloader,
        iterator,
        extractor,
        output_format,
        output_type=output_type,
        keep_raw_download=keep_raw_download,
        force_download=force_download,
    )

    return dataset
