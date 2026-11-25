import json
import sys
from collections import Counter
from unittest.mock import patch

import pytest

from tools.postprocess.data_mixture import parse_args, run_mixture


@pytest.fixture(params=["source", "source with spaces"])
def sources(tmp_path, request):
    paths = []
    for source in ("a", "b"):
        path = tmp_path / f"{request.param}_{source}.jsonl"
        path.write_text(
            "".join(json.dumps({"text": f"{source}-{i}", "source": source}) + "\n" for i in range(12)),
            encoding="utf-8",
        )
        paths.append(path.as_posix())
    return paths


@pytest.mark.parametrize(
    "budget,expected",
    [(None, {"a": 12, "b": 12}), (8, {"a": 2, "b": 6}), (20, {"a": 5, "b": 15})],
)
def test_export_weighted_mixture(tmp_path, sources, budget, expected):
    output = tmp_path / "mixed.jsonl"
    argv = [
        "data_mixture.py",
        "--data_path",
        "1",
        sources[0],
        "3",
        sources[1],
        "--export_path",
        str(output),
        "--num_proc",
        "1",
    ]
    if budget is not None:
        argv.extend(["--max_samples", str(budget)])
    with patch.object(sys, "argv", argv):
        run_mixture()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert Counter(row["source"] for row in rows) == expected
    assert all(row["text"] in {f'{row["source"]}-{i}' for i in range(12)} for row in rows)


@pytest.mark.parametrize("budget", ["0", "-1"])
def test_reject_nonpositive_sample_budget(budget, capsys):
    with patch.object(sys, "argv", ["data_mixture.py", "--data_path", "unused.jsonl", "--max_samples", budget]):
        with pytest.raises(SystemExit) as error:
            parse_args()
    assert error.value.code == 2
    assert "--max_samples must be a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("args", [[], ["--data_path"]])
def test_require_data_paths(args):
    with patch.object(sys, "argv", ["data_mixture.py", *args]):
        with pytest.raises(SystemExit) as error:
            parse_args()
    assert error.value.code == 2
