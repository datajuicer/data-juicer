from dataclasses import dataclass, field
from typing import Dict, List, Optional

import regex as re
from loguru import logger

from ..base_op import OPERATORS, Mapper

OP_NAME = "latex_figure_context_extractor_mapper"


@dataclass
class SubFigure:
    """A subfigure within a figure environment."""

    caption: str = ""
    label: str = ""
    image_paths: List[str] = field(default_factory=list)


@dataclass
class Figure:
    """A top-level figure/figure* environment."""

    caption: str = ""
    label: str = ""
    image_paths: List[str] = field(default_factory=list)
    sub_figures: List[SubFigure] = field(default_factory=list)


@OPERATORS.register_module(OP_NAME)
class LatexFigureContextExtractorMapper(Mapper):
    """Extracts figures and their citing context from LaTeX source.

    This operator parses figure environments from a paper's LaTeX
    source, extracts each figure's caption, label, and image path(s),
    and finds the prose paragraphs that cite each figure.  It fans out
    one paper row into N figure rows (one per figure or subfigure).
    **Samples that contain no figures with images are dropped from
    the output.**

    Supported figure environments: figure, figure*, wrapfigure,
        subfigure (environment), \\subfigure (command),
        \\subfloat (command, subfig package).
    Supported caption commands: \\caption, \\caption*,
        \\subcaption, \\captionof{figure}.

    Figures without \\includegraphics are skipped.  Subfigures
    inherit citing paragraphs from their parent figure's label.

    Output fields (in addition to all input fields):

    - ``<image_key>`` (default ``images``, inherited from base
      class): list of image paths from ``\\includegraphics``.
    - ``<caption_key>`` (default ``caption``): figure caption text.
    - ``<label_key>`` (default ``label``): LaTeX label string.
    - ``<context_key>`` (default ``citing_paragraphs``): list of
      paragraphs that cite this figure.
    - ``<parent_caption_key>`` (default ``parent_caption``): parent
      figure caption (subfigures only; empty for standalone figures).
    - ``<parent_label_key>`` (default ``parent_label``): parent
      figure label (subfigures only; empty for standalone figures).

    Note: this operator expects the full LaTeX source as a single
    string.  It does **not** resolve ``\\input`` or ``\\include``
    directives.  If your documents span multiple ``.tex`` files,
    concatenate them into a single text field before applying this
    mapper.
    """

    _batched_op = True

    # Recursive nested braces pattern via the ``regex`` module.
    # Matches balanced ``{...}`` content at arbitrary nesting depth.
    # ``(?P<B>...)`` defines a named group for a single balanced brace
    # pair; ``(?&B)`` recurses into it, so
    # ``\caption{A \textbf{B \emph{C \footnote{D \cite{E}}}}}``
    # is matched correctly regardless of depth.
    _NESTED_BRACES = r"(?:[^{}]|(?P<B>\{(?:[^{}]|(?&B))*\}))*"

    # LaTeX environments stripped from prose before searching for
    # citing paragraphs.  Only float / display / verbatim environments
    # are listed — structural ones (document, section, itemize, …)
    # are kept so that prose inside them is still searchable.
    _STRIPPED_ENVS = (
        "figure",
        r"figure\*",
        "wrapfigure",
        "table",
        r"table\*",
        "tabular",
        r"tabular\*",
        "equation",
        r"equation\*",
        "align",
        r"align\*",
        "alignat",
        r"alignat\*",
        "gather",
        r"gather\*",
        "multline",
        r"multline\*",
        "flalign",
        r"flalign\*",
        "algorithm",
        r"algorithm\*",
        "lstlisting",
        "verbatim",
        "minted",
    )

    def __init__(
        self,
        citation_commands: Optional[List[str]] = None,
        paragraph_separator: str = "\n\n",
        caption_key: str = "caption",
        label_key: str = "label",
        context_key: str = "citing_paragraphs",
        parent_caption_key: str = "parent_caption",
        parent_label_key: str = "parent_label",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param citation_commands: LaTeX reference commands to search
            for when finding citing paragraphs. Defaults to
            ['\\ref', '\\cref', '\\Cref', '\\autoref'].
            Comma-separated label lists (e.g. ``\\cref{fig:a,fig:b}``)
            are handled automatically.
        :param paragraph_separator: Pattern for splitting LaTeX text
            into paragraphs. Defaults to '\\n\\n'.
        :param caption_key: Output field name for the figure caption.
        :param label_key: Output field name for the LaTeX label.
        :param context_key: Output field name for citing paragraphs.
        :param parent_caption_key: Output field name for the parent
            figure's caption. For subfigures this carries the parent
            figure environment's caption; for standalone figures it
            is an empty string.
        :param parent_label_key: Output field name for the parent
            figure's label. Useful for grouping subfigures that
            belong to the same figure environment. Empty string for
            standalone figures.
        :param args: extra args
        :param kwargs: extra args.  Notably ``text_key`` (default
            ``'text'``) controls which input field contains the LaTeX
            source, and ``image_key`` (default ``'images'``) controls
            the output field name for extracted image paths.  Both
            are inherited from the base ``OP`` class.
        """
        super().__init__(*args, **kwargs)
        if citation_commands is None:
            citation_commands = [
                r"\ref",
                r"\cref",
                r"\Cref",
                r"\autoref",
            ]
        self.citation_commands = citation_commands
        self.paragraph_separator = paragraph_separator

        # Pre-build the citation command alternation once (it never
        # changes after init).  Individual label patterns are cached
        # lazily in _citation_pattern_cache.
        cmd_names = [cmd.lstrip("\\") for cmd in citation_commands]
        self._cite_cmd_alt = "|".join(re.escape(c) for c in cmd_names)
        self._citation_pattern_cache: Dict[str, re.Pattern] = {}
        self.caption_key = caption_key
        self.label_key = label_key
        self.context_key = context_key
        self.parent_caption_key = parent_caption_key
        self.parent_label_key = parent_label_key

        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns used by the parser.

        Called once from ``__init__``.  Separated for readability —
        the patterns are non-trivial and benefit from being grouped
        together away from the parameter-handling logic.
        """
        nb = self._NESTED_BRACES

        # -- Figure environments ------------------------------------------
        # figure, figure*, wrapfigure (with optional {pos}{width} args).
        # Named group + backreference so \begin{X} only matches \end{X}.
        self._figure_env_pattern = re.compile(
            r"\\begin\{(?P<fig_env>figure\*?|wrapfigure)\}"
            r"(?:\[[^\]]*\])*"  # skip optional [...] args
            r"(?:\{[^}]*\})*"  # skip mandatory {...} args
            r".*?"
            r"\\end\{(?P=fig_env)\}",
            re.DOTALL,
        )

        # -- Subfigure environments ---------------------------------------
        # \begin{subfigure}[pos]{width}...\end{subfigure}
        self._subfigure_env_pattern = re.compile(
            r"\\begin\{subfigure\}"
            r"(?:\[[^\]]*\])*"  # optional [pos] arg
            r"(?:\{[^}]*\})*"  # mandatory {width} arg
            r".*?"
            r"\\end\{subfigure\}",
            re.DOTALL,
        )

        # -- Subfigure / subfloat commands --------------------------------
        # \subfigure[caption]{content} or \subfloat[caption]{content}
        self._subfigure_cmd_pattern = re.compile(r"\\(?:subfigure|subfloat)\[([^\]]*)\]" r"\s*" r"\{(" + nb + r")\}")
        # \subfigure{content} or \subfloat{content} (no optional caption)
        self._subfig_cmd_nocaption_pattern = re.compile(
            r"\\(?:subfigure|subfloat)" r"(?!\[)" r"\s*" r"\{(" + nb + r")\}"
        )

        # -- Caption commands ---------------------------------------------
        self._caption_pattern = re.compile(r"\\caption\*?(?:\[[^\]]*\])?\{(" + nb + r")\}")
        self._subcaption_pattern = re.compile(r"\\subcaption(?:\[[^\]]*\])?\{(" + nb + r")\}")
        self._captionof_pattern = re.compile(r"\\captionof\{figure\}(?:\[[^\]]*\])?\{(" + nb + r")\}")
        # \captionof{table}{...} — used to detect table minipages inside
        # figure environments so they can be excluded from figure output.
        self._captionof_table_pattern = re.compile(r"\\captionof\{table\}(?:\[[^\]]*\])?\{(" + nb + r")\}")

        # -- Minipage environments ----------------------------------------
        # Matches \begin{minipage}[pos]{width}...\end{minipage}.
        self._minipage_pattern = re.compile(
            r"\\begin\{minipage\}" r"(?:\[[^\]]*\])*" r"(?:\{[^}]*\})*" r"(.*?)" r"\\end\{minipage\}",
            re.DOTALL,
        )

        # -- Label and includegraphics ------------------------------------
        self._label_pattern = re.compile(r"\\label\{([^}]+)\}")
        self._includegraphics_pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")

        # -- Environment stripping ----------------------------------------
        # Removes float/display/verbatim environments so that
        # citing-paragraph search only sees prose text.
        env_alt = "|".join(self._STRIPPED_ENVS)
        self._env_strip_pattern = re.compile(
            r"\\begin\{(" + env_alt + r")\}" r".*?" r"\\end\{\1\}",
            re.DOTALL,
        )

    def _extract_caption(self, text):
        """Extract caption text from a LaTeX fragment.

        Tries \\caption (including \\caption*), \\subcaption, and
        \\captionof{figure}.  Returns the first match's content,
        or ''.
        """
        for pattern in (self._caption_pattern, self._subcaption_pattern, self._captionof_pattern):
            m = pattern.search(text)
            if m:
                return m.group(1).strip()
        return ""

    def _extract_label(self, text):
        """Extract the first \\label{...} value from a LaTeX
        fragment."""
        m = self._label_pattern.search(text)
        return m.group(1).strip() if m else ""

    def _extract_image_paths(self, text):
        """Extract all \\includegraphics paths from a LaTeX
        fragment."""
        return [m.group(1).strip() for m in self._includegraphics_pattern.finditer(text)]

    def _build_subfigure(self, caption, content):
        """Build a SubFigure from an explicit caption and content text.

        :param caption: the caption string (already extracted).
        :param content: LaTeX fragment to extract label and image
            paths from.
        :return: a SubFigure instance.
        """
        return SubFigure(
            caption=caption,
            label=self._extract_label(content),
            image_paths=self._extract_image_paths(content),
        )

    def _is_table_minipage(self, text):
        """Check if a minipage contains a \\captionof{table} command,
        indicating it holds a table rather than a figure.

        :param text: the minipage body text.
        :return: True if the minipage is a table.
        """
        return bool(self._captionof_table_pattern.search(text))

    def _parse_figure_env(self, fig_text):
        """Parse a figure/figure*/wrapfigure environment block.

        Handles \\begin{subfigure} environments,
        \\subfigure[caption]{content} commands (older subfigure
        package), and \\subfloat[caption]{content} commands (subfig
        package).  Commands without the optional [caption] argument
        are also supported.

        Also handles the minipage pattern where a single figure
        environment contains multiple \\begin{minipage} blocks, each
        with its own \\caption, \\label, and \\includegraphics.
        Minipages that contain \\captionof{table} are skipped.

        :param fig_text: the full text of a figure environment.
        :return: a Figure object, a list of Figure objects, or None
            if it has no images.
        """
        # Check for \begin{subfigure}...\end{subfigure} environments
        subfig_env_matches = list(self._subfigure_env_pattern.finditer(fig_text))
        # Check for \subfigure[caption]{} / \subfloat[caption]{} commands
        subfig_cmd_matches = list(self._subfigure_cmd_pattern.finditer(fig_text))
        # Check for \subfigure{} / \subfloat{} commands (no caption)
        subfig_nocap_matches = list(self._subfig_cmd_nocaption_pattern.finditer(fig_text))

        has_subfigures = bool(subfig_env_matches or subfig_cmd_matches or subfig_nocap_matches)

        if has_subfigures:
            # Normalise every subfigure variant into (caption, content)
            # so we can parse them in a single pass.
            caption_content_pairs = []
            for m in subfig_env_matches:
                # \begin{subfigure}...\end{subfigure}: caption is
                # inside the environment body, content is the whole match
                caption_content_pairs.append((self._extract_caption(m.group(0)), m.group(0)))
            for m in subfig_cmd_matches:
                # \subfigure[caption]{content} / \subfloat[caption]{content}
                caption_content_pairs.append((m.group(1).strip(), m.group(2)))
            for m in subfig_nocap_matches:
                # \subfigure{content} / \subfloat{content} (no caption)
                caption_content_pairs.append(("", m.group(1)))

            sub_figures = []
            for caption, content in caption_content_pairs:
                sf = self._build_subfigure(caption, content)
                if sf.image_paths:
                    sub_figures.append(sf)

            # Extract parent caption/label from text outside
            # all subfigure/subfloat blocks
            text_outside = fig_text
            all_matches = sorted(
                subfig_env_matches + subfig_cmd_matches + subfig_nocap_matches,
                key=lambda m: m.start(),
                reverse=True,
            )
            for m in all_matches:
                text_outside = text_outside[: m.start()] + text_outside[m.end() :]

            if not sub_figures:
                return None

            return Figure(
                caption=self._extract_caption(text_outside),
                label=self._extract_label(text_outside),
                image_paths=[],
                sub_figures=sub_figures,
            )
        else:
            # No subfigures — check for the minipage pattern.
            # When a figure environment contains multiple minipages
            # with separate \caption/\label pairs, each minipage is
            # treated as an independent figure.  Minipages holding
            # \captionof{table} are skipped (they are tables).
            minipage_matches = list(self._minipage_pattern.finditer(fig_text))
            if len(minipage_matches) >= 2:
                figures = []
                for mp in minipage_matches:
                    mp_body = mp.group(1)
                    # Skip table minipages
                    if self._is_table_minipage(mp_body):
                        continue
                    image_paths = self._extract_image_paths(mp_body)
                    if not image_paths:
                        continue
                    figures.append(
                        Figure(
                            caption=self._extract_caption(mp_body),
                            label=self._extract_label(mp_body),
                            image_paths=image_paths,
                            sub_figures=[],
                        )
                    )
                return figures if figures else None

            # Single figure (no subfigures, no multi-minipage)
            image_paths = self._extract_image_paths(fig_text)
            if not image_paths:
                return None

            return Figure(
                caption=self._extract_caption(fig_text),
                label=self._extract_label(fig_text),
                image_paths=image_paths,
                sub_figures=[],
            )

    def _parse_figures(self, latex_source):
        """Parse all figure environments from a LaTeX source.

        :param latex_source: full LaTeX document source.
        :return: a list of Figure objects.
        """
        figures = []
        for m in self._figure_env_pattern.finditer(latex_source):
            result = self._parse_figure_env(m.group(0))
            if result is None:
                continue
            if isinstance(result, list):
                figures.extend(result)
            else:
                figures.append(result)
        return figures

    def _prepare_paragraphs(self, latex_source):
        """Strip float/display environments and split into clean
        paragraphs.

        :param latex_source: full LaTeX document source.
        :return: a list of non-empty paragraph strings.
        """
        stripped = self._env_strip_pattern.sub("", latex_source)
        return [p.strip() for p in stripped.split(self.paragraph_separator) if p.strip()]

    def _get_citation_pattern(self, label):
        """Return a compiled regex that matches any citation command
        referencing *label*.  Results are cached per label.

        Handles comma-separated label lists such as
        ``\\cref{fig:a,fig:b}``.  The label must appear as a
        complete entry (bounded by ``{``, ``,``, or ``}``) so that
        e.g. label ``fig:a`` does not false-match ``fig:ab``.

        :param label: the LaTeX label string to search for.
        :return: a compiled regex pattern (cached).
        """
        pat = self._citation_pattern_cache.get(label)
        if pat is not None:
            return pat
        # Match \cmd{...label...} where label appears as a
        # complete entry in a comma-separated list.
        # (?:[^},]*,\s*)*  — zero or more preceding entries
        # LABEL             — the target label
        # \s*(?:,[^}]*)?    — optional trailing entries
        pat = re.compile(
            r"\\(?:" + self._cite_cmd_alt + r")\{" r"(?:[^},]*,\s*)*" + re.escape(label) + r"\s*(?:,[^}]*)?\}"
        )
        self._citation_pattern_cache[label] = pat
        return pat

    def _find_citing_paragraphs(self, label, paragraphs):
        """Find paragraphs that cite a given label.

        :param label: the LaTeX label to search for.
        :param paragraphs: list of paragraph strings to search in.
        :return: a list of paragraph strings that cite the label,
            or [] if label is empty.
        """
        if not label:
            return []
        cite_pattern = self._get_citation_pattern(label)
        return [p for p in paragraphs if cite_pattern.search(p)]

    def _append_output_row(self, output_samples, samples, idx, input_keys, *, fig, citing_paragraphs, parent=None):
        """Append one output row to the output_samples dict.

        :param output_samples: accumulator dict of lists.
        :param samples: the original input batch.
        :param idx: index of the current sample in the batch.
        :param input_keys: keys to copy from the original sample.
        :param fig: a Figure or SubFigure whose caption, label,
            and image_paths are emitted.
        :param citing_paragraphs: list of citing paragraph strings.
        :param parent: optional parent Figure for subfigure rows.
            When provided, its caption and label are emitted as
            parent_caption / parent_label; otherwise empty strings.
        """
        # Keys that are explicitly set below must be skipped during
        # the input-copy loop to avoid double-appending (e.g. when
        # the input batch already contains an ``images`` column).
        output_only_keys = {
            self.image_key,
            self.caption_key,
            self.label_key,
            self.context_key,
            self.parent_caption_key,
            self.parent_label_key,
        }
        for k in input_keys:
            if k not in output_only_keys:
                output_samples[k].append(samples[k][idx])
        output_samples[self.caption_key].append(fig.caption)
        output_samples[self.image_key].append(fig.image_paths)
        output_samples[self.label_key].append(fig.label)
        output_samples[self.context_key].append(citing_paragraphs)
        output_samples[self.parent_caption_key].append(parent.caption if parent else "")
        output_samples[self.parent_label_key].append(parent.label if parent else "")

    def process_batched(self, samples):
        input_keys = samples.keys()
        num_samples = len(samples[next(iter(input_keys))])
        output_keys = input_keys | {
            self.caption_key,
            self.label_key,
            self.context_key,
            self.image_key,
            self.parent_caption_key,
            self.parent_label_key,
        }
        output_samples: Dict[str, list] = {key: [] for key in output_keys}

        for i in range(num_samples):
            latex_source = samples[self.text_key][i]

            # Parse figures
            figures = self._parse_figures(latex_source)

            if not figures:
                logger.warning(
                    f"No figures with images found in sample {i} "
                    f"(batch size {num_samples}). "
                    f"Sample will be dropped from output."
                )
                continue

            # Prepare cleaned paragraphs once per paper
            paragraphs = self._prepare_paragraphs(latex_source)

            # Fan out
            for fig in figures:
                if fig.sub_figures:
                    # Parent-level citing paragraphs
                    parent_citing = self._find_citing_paragraphs(fig.label, paragraphs)

                    for sf in fig.sub_figures:
                        # Subfigure-specific citing paragraphs
                        sf_citing = self._find_citing_paragraphs(sf.label, paragraphs)
                        # Merge parent + subfigure, deduplicated,
                        # preserving order
                        merged = list(dict.fromkeys(parent_citing + sf_citing))

                        self._append_output_row(
                            output_samples,
                            samples,
                            i,
                            input_keys,
                            fig=sf,
                            citing_paragraphs=merged,
                            parent=fig,
                        )
                else:
                    # Single figure (leaf) — no parent
                    citing = self._find_citing_paragraphs(fig.label, paragraphs)
                    self._append_output_row(
                        output_samples,
                        samples,
                        i,
                        input_keys,
                        fig=fig,
                        citing_paragraphs=citing,
                    )

        return output_samples
