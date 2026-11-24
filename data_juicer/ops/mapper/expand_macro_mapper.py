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

    # regex for extracting \newcommand macros without arguments
    _non_arg_nc_reg = re.compile(
        # this regex matches the following:
        # \newcommand{\macro_name}{macro_value}
        # \newcommand*{\macro_name}{macro_value}
        # where macro_name is only allowed to contain letters and numbers;
        # macro_value can contain any character.
        pattern=r"\\\bnewcommand\b\*?\{(\\[a-zA-Z0-9]+?)\}\{(.*?)\}$",
        flags=re.MULTILINE,
    )

    # regex for extracting \def macros without arguments
    _non_arg_def_reg = re.compile(
        # this regex matches the following:
        # \def\macro_name{macro_value}
        # where macro_name is only allowed to contain letters and numbers;
        # macro_value can contain any character.
        pattern=r"\\def\s*(\\[a-zA-Z0-9]+?)\s*\{(.*?)\}$",
        flags=re.MULTILINE,
    )

    # regex for splitting a text into the spans that macro expansion can
    # change: a span starts at a backslash and covers all directly following
    # letters, numbers and backslashes. A macro match never reaches out of
    # its span, so every span is expanded on its own.
    _macro_span_reg = re.compile(r"(\\[a-zA-Z0-9\\]*)")

    # regex for splitting a text around its control words. A control word
    # covers the whole alphanumeric run, so a macro that equals a control
    # word is never a part of a longer alphanumeric word; macros at the end
    # of the text or directly followed by another macro are found too.
    _control_word_reg = re.compile(r"(\\[a-zA-Z0-9]+)")

    # regex for the empty match at the start of a text. An escaped macro
    # value is a re.sub template; a substitution of this match in an empty
    # text gives the text that the macro inserts.
    _text_start_reg = re.compile("^")

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

    def _build_non_arg_macros_dict(self, file_content):
        # Extract all user-defined LaTeX macros from the preamble
        macros = {}
        for reg in [self._non_arg_nc_reg, self._non_arg_def_reg]:
            for match in reg.finditer(file_content):
                # convert the macro name and value to a raw string that can be
                # used in re.sub
                macro_name = match.group(1).encode("unicode-escape").decode("utf-8")
                macro_val = match.group(2).encode("unicode-escape").decode("utf-8")

                macros[macro_name] = macro_val
        return macros

    def _expand_non_arg_macros(self, text, macros, positions):
        """
        Inline-expand the macros in the text one after another in
        definition order.

        :param text: text to expand
        :param macros: list of (control word, replacement text) in
            definition order
        :param positions: position in `macros` of each control word
        :return: expanded text
        """
        next_pos = 0
        while True:
            # the control words are the odd items of the split result
            parts = self._control_word_reg.split(text)
            # only the macros that occur in the text need a substitution;
            # macros before `next_pos` had their turn already
            pending = [positions[word] for word in parts[1::2] if positions.get(word, -1) >= next_pos]
            if not pending:
                return text
            pos = min(pending)
            control_word, replacement = macros[pos]
            parts[1::2] = [replacement if word == control_word else word for word in parts[1::2]]
            text = "".join(parts)
            next_pos = pos + 1

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            non_arg_macros = self._build_non_arg_macros_dict(text)

            # TODO: macros that take arguments are not supported yet
            arg_macros = {}

            # inline-expand all non-arg macros
            if non_arg_macros:
                # the escaped macro name only adds a second leading backslash
                # to the control word; the escaped value is a re.sub
                # template, so let re.sub make the text that it inserts
                macros = [
                    (macro_name[1:], self._text_start_reg.sub(repl=macro_value, string="", count=1))
                    for macro_name, macro_value in non_arg_macros.items()
                ]
                positions = {control_word: pos for pos, (control_word, _) in enumerate(macros)}
                # expand each distinct span once and reuse the result
                parts = self._macro_span_reg.split(text)
                expanded = {span: self._expand_non_arg_macros(span, macros, positions) for span in set(parts[1::2])}
                parts[1::2] = map(expanded.__getitem__, parts[1::2])
                text = "".join(parts)

            # inline-expand all macros that use args
            # TODO: inline-expand macros with args
            for macro_name, macro_value in arg_macros.items():
                pass

            samples[self.text_key][idx] = text

        return samples
