import re

MD_FLAGS = re.MULTILINE | re.DOTALL


def parse_existing_op_md(md_text: str) -> dict:
    """
    Parse Markdown previously rendered by op_doc.md.j2 and return structured data.

    Returns:
      {
        "name": str,
        "desc": str,
        "type": str,
        "tags": [str],
        "params": [{"name":..,"type":..,"default":..,"desc":..}, ...],
        "examples": [
          {"method": str, "op_code": str, "input": str, "output": str, "explanation": str},
          ...
        ],
        "code_links": {"source": str, "test": str}
      }

    Missing sections yield empty values.
    """
    res = {
        "name": "",
        "desc": "",
        "type": "",
        "tags": [],
        "params": [],
        "examples": dict(),
        "code_links": {"source": "", "test": ""},
    }
    text = md_text

    # Title and description
    m = re.search(r"^\#\s+(?P<name>.+?)\s*$", text, flags=MD_FLAGS)
    if m:
        res["name"] = m.group("name").strip()
        title_end = m.end()
    else:
        title_end = 0

    m_type = re.search(r"^Type\s*[^:]*:\s*\*\*(?P<type>.+?)\*\*\s*$", text, flags=MD_FLAGS)
    type_start = m_type.start() if m_type else len(text)
    if title_end < type_start:
        desc = text[title_end:type_start].strip()
        res["desc"] = desc.strip()
    if m_type:
        res["type"] = m_type.group("type").strip()

    # Tags line
    m_tag = re.search(r"^Tags\s*标签:\s*(?P<tags>.+?)\s*$", text, flags=MD_FLAGS)
    if m_tag:
        tags_line = m_tag.group("tags").strip()
        if tags_line:
            res["tags"] = [t.strip() for t in tags_line.split(",") if t.strip()]

    # Parameter table
    m_param_sec = re.search(r"^\#\#\s*🔧\s*Parameter Configuration.*?$", text, flags=MD_FLAGS)
    if m_param_sec:
        rows = []
        m_next = re.search(r"^\#\#\s*📊\s*Effect demonstration.*?$", text, flags=MD_FLAGS)
        sec_text = text[m_param_sec.end() : m_next.start() if m_next else len(text)]
        for line in sec_text.splitlines():
            line = line.rstrip()
            mrow = re.match(
                r"^\|\s*`(?P<name>[^`]+)`\s*\|\s*(?P<type>[^|]+?)\s*\|\s*`(?P<default>[^`]*)`\s*\|\s*(?P<desc>.*?)\s*\|$",
                line,
            )
            if mrow:
                rows.append(
                    {
                        "name": mrow.group("name").strip(),
                        "type": mrow.group("type").strip(),
                        "default": mrow.group("default").strip(),
                        "desc": mrow.group("desc").strip(),
                    }
                )
        res["params"] = rows

    # Examples section
    m_effect = re.search(r"^\#\#\s*📊\s*Effect demonstration.*?$", text, flags=MD_FLAGS)
    m_links = re.search(r"^\#\#\s*🔗\s*related links.*?$", text, flags=MD_FLAGS)
    if m_effect:
        sec = text[m_effect.end() : m_links.start() if m_links else len(text)]
        if "not available" not in sec and "暂无" not in sec:
            blocks = []
            for mth in re.finditer(r"^\#\#\#\s+(?P<method>.+?)\s*$", sec, flags=MD_FLAGS):
                blocks.append((mth.group("method").strip(), mth.start(), mth.end()))
            for i, (method, s, e) in enumerate(blocks):
                end = blocks[i + 1][1] if i + 1 < len(blocks) else len(sec)
                b = sec[e:end]

                # Optional operator code block
                op_code = ""
                mcode = re.search(r"```python\s*(?P<code>.*?)```", b, flags=MD_FLAGS)
                if mcode:
                    op_code = mcode.group("code").strip()

                # Input/output/explanation subsections
                minput = re.search(r"^####\s*📥\s*input data.*?$", b, flags=MD_FLAGS)
                moutput = re.search(r"^####\s*📤\s*output data.*?$", b, flags=MD_FLAGS)
                mexpl = re.search(r"^####\s*✨\s*explanation.*?$", b, flags=MD_FLAGS)

                input_text = output_text = explanation = ""
                if minput and moutput:
                    input_text = b[minput.end() : moutput.start()].strip()
                if moutput and mexpl:
                    output_text = b[moutput.end() : mexpl.start()].strip()
                elif moutput:
                    output_text = b[moutput.end() :].strip()

                if mexpl:
                    explanation = b[mexpl.end() :].strip()
                    explanation = explanation.strip() if "TODO" not in explanation else ""

                res["examples"][method] = {
                    "method": method,
                    "op_code": op_code,
                    "input": input_text,
                    "output": output_text,
                    "explanation": explanation,
                }

    # Related links
    if m_links:
        link_sec = text[m_links.end() :]
        m_src = re.search(r"\[source code.*?\]\((?P<src>[^)]+)\)", link_sec)
        m_tst = re.search(r"\[unit test.*?\]\((?P<test>[^)]+)\)", link_sec)
        res["code_links"]["source"] = (m_src.group("src") if m_src else "").strip()
        res["code_links"]["test"] = (m_tst.group("test") if m_tst else "").strip()

    return res


def load_existing_op_md(path):
    """Load and parse existing operator markdown if present."""
    if path.exists():
        try:
            return parse_existing_op_md(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[MD parse] {path} parse failed: {e}")
    return None
