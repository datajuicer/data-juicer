# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/blob/rp_v1/data_prep/arxiv/arxiv_cleaner.py
# --------------------------------------------------------

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("expand_macro_mapper")
class ExpandMacroMapper(Mapper):
    """Expands macro definitions in the document body of LaTeX samples.

    This operator processes LaTeX documents to expand user-defined macros in the text. It
    supports \\newcommand and \\def macros without arguments. Macros are identified and
    expanded in the text, ensuring they are not part of longer alphanumeric words. The
    operator currently does not support macros with arguments. The processed text is updated
    in the samples."""

    _batched_op = True

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

    def _build_non_arg_macros_dict(self, file_content):
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

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            non_arg_macros = self._build_non_arg_macros_dict(text)

            # TODO: macros that take arguments are not supported yet
            arg_macros = {}

            # inline-expand all non-arg macros
            for macro_name, macro_value in non_arg_macros.items():
                text = re.sub(
                    # make pattern grouped to make sure that the macro
                    # is not part of a longer alphanumeric word
                    pattern=r"(" + macro_name + r")" + r"([^a-zA-Z0-9])",
                    # replace the macro with its value and add back the
                    # character that was matched after the macro
                    repl=macro_value + r"\2",
                    string=text,
                )

            # inline-expand all macros that use args
            # TODO: inline-expand macros with args
            for macro_name, macro_value in arg_macros.items():
                pass

            samples[self.text_key][idx] = text

        return samples
