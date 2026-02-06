from utils.model import chat
from utils.prompts import PROMPTS
import json

PROMPT_BRIEF_DELIM = "\n\n-----\n\n"

def get_bilingual_descs(descs):
    """
    Translate a list of English descriptions to Chinese in batches using LLM
    and return merged bilingual descriptions. Batch split is based on length.
    Terminology rules:
      - operator -> 算子
      - Hugging Face -> keep English
      - token -> keep English
    """
    if not descs:
        return []
    separator = "\n\n******\n\n"
    limit = int(5e3)

    def _translate_batch_with_llm(batch_text, sep):
        system = PROMPTS.get("translate_system") or (
            "You are a professional bilingual technical translator (English -> Simplified Chinese)."
        )
        user = PROMPTS.get("translate_user_template", "Translate with separator {separator}:\n{batch}").format(
            separator=sep,
            batch=batch_text,
        )

        retry = 0
        last_err = None
        while retry < 3:
            try:
                content = chat(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                )
                if isinstance(content, str):
                    return content
                else:
                    last_err = RuntimeError(f"Unexpected response type: {type(content)}, content={content}")
            except Exception as e:
                last_err = e
                print(f"[LLM translate] error: {e} retry={retry} len_batch={len(batch_text)}")
            retry += 1
        print(f"[LLM translate] failed after retries. last_err={last_err}")
        return None

    batch = separator.join(descs)
    if len(batch) > limit:
        split_idx = int(len(descs) / 2)
        res1 = get_bilingual_descs(descs[:split_idx])
        res2 = get_bilingual_descs(descs[split_idx:])
        return res1 + res2
    else:
        res = _translate_batch_with_llm(batch, separator)

    if not res:
        zhs = [""] * len(descs)
    else:
        print(f"[LLM Translate] ori_len: {len(batch)}")
        zhs = res.split(separator)
        assert len(zhs) == len(descs)

    assert len(zhs) == len(descs)
    return [desc + "\n\n" + zh.strip() for desc, zh in zip(descs, zhs)]


def _build_example_brief(method, vals):
    """
    Build a compact text brief for a test example to feed into LLM prompts.
    vals keys: op_code, ds, tgt, samples, test_code.
    """
    parts = [f"method: {method}", "Here is what is shown to users (If this method is selected):"]
    if vals.op_code:
        parts.append(f"Code that executes this operator: {vals.op_code}")
    if vals.input:
        parts.append(f"input data: {vals.input}")
    if vals.output:
        parts.append(f"output data: {vals.output}")
    brief = "\n".join(parts)
    if len(brief) > 2000:
        return ""
    return brief


def select_and_explain_examples(examples, op_info, test_file_full, skip_explain=False):
    """
    Drive LLM to select and explain examples.

    Behavior:
    - Always provide the FULL test file content and the list of pre-screened method names.
    - If len(pre-screened) < 2, the prompt instructs the model to select all of them.
    - The model must return JSON with 'selected' and 'explanations'.

    Returns: (selected_methods: List[str], explanations: Dict[method, str])
    """
    if not examples:
        return [], {}

    if skip_explain:
        selected = list(examples.keys())[:2]
        explanations = {m: "" for m in selected}
        return selected, explanations

    briefs = []
    op_desc = ""
    if op_info.get("desc"):
        op_desc = op_info.get("desc")
        briefs.append(f"op_desc: {op_desc}")
    for m, v in examples.items():
        briefs.append(_build_example_brief(m, v))

    briefs_string = PROMPT_BRIEF_DELIM.join(briefs)
    if len(briefs_string) > 5000:
        briefs_string = briefs_string[:5000] + "..." + "subsequent omission"

    methods_all = list(examples.keys())

    system = PROMPTS["select_system"]
    user = PROMPTS["select_user_template"].format(
        test_file_full=test_file_full,
        briefs=briefs_string,
        json_example=PROMPTS["select_json_example"],
    )

    try:
        resp = chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        data = resp if isinstance(resp, dict) else json.loads(str(resp))
        # Only keep methods that are in the pre-screened list; cap at k
        selected = [m for m in (data.get("selected") or []) if m in examples]
        if len(methods_all) < 2:
            # If fewer than 2 pre-screened, ensure we use all of them
            selected = methods_all
        else:
            selected = selected[:2] if selected else methods_all[:2]
        expl = data.get("explanations") or {}
        explanations = {m: str(expl.get(m, "")).strip() for m in selected}
        print(f"[LLM select+explain] selected: {selected}, explanations: {list(explanations.values())}")
        return selected, explanations
    except Exception as e:
        print(f"[LLM select+explain] parse error: {e} op: {op_info['name']}")
        return [], {}
