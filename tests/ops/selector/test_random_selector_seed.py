import numpy as np
import pytest
import yaml
from datasets import Dataset

from data_juicer.config import init_configs
from data_juicer.ops.load import load_ops
from data_juicer.ops.selector.random_selector import RandomSelector


@pytest.fixture
def dataset():
    return Dataset.from_dict({"id": list(range(100)), "text": [f"sample {i}" for i in range(100)]})


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("indexed", [False, True])
def test_seed_matches_dataset_sampling(dataset, seed, indexed):
    if indexed:
        dataset = dataset.select(list(reversed(range(100))))
    expected = dataset.shuffle(seed=seed).select(range(17)).to_list()
    op = RandomSelector(select_num=17, seed=seed)
    assert op.process(dataset).to_list() == expected
    assert op.process(dataset).to_list() == expected


def test_default_preserves_previous_seed(dataset):
    expected = dataset.shuffle(seed=42).select(range(17)).to_list()
    assert RandomSelector(select_num=17).process(dataset).to_list() == expected


def test_recipe_seed_changes_selected_rows(dataset):
    recipe = yaml.safe_load("process:\n  - random_selector:\n      select_num: 17\n      seed: 7\n")
    op = load_ops(recipe["process"])[0]
    selected = op.process(dataset).to_list()
    assert selected == dataset.shuffle(seed=7).select(range(17)).to_list()
    assert selected != RandomSelector(select_num=17, seed=42).process(dataset).to_list()


def test_cli_overrides_recipe_seed(tmp_path, dataset):
    source = tmp_path / "input.jsonl"
    source.write_text('{"text": "sample"}\n', encoding="utf-8")
    config = tmp_path / "recipe.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset_path": source.as_posix(),
                "export_path": (tmp_path / "output.jsonl").as_posix(),
                "process": [{"random_selector": {"select_num": 17, "seed": 42}}],
            }
        ),
        encoding="utf-8",
    )
    cfg = init_configs(args=["--config", str(config), "--random_selector.seed", "7"], load_configs_only=True)
    op = load_ops(cfg.process)[0]
    assert op.process(dataset).to_list() == dataset.shuffle(seed=7).select(range(17)).to_list()


@pytest.mark.parametrize("config,count", [({"select_ratio": 0.25}, 25), ({"select_ratio": 0.25, "select_num": 7}, 7)])
def test_ratio_and_count_use_seed(dataset, config, count):
    expected = dataset.shuffle(seed=7).select(range(count)).to_list()
    assert RandomSelector(seed=7, **config).process(dataset).to_list() == expected


def test_seeded_oversampling(dataset):
    expected = dataset.shuffle(seed=7).select(list(range(100)) + list(range(25))).to_list()
    assert RandomSelector(select_num=125, seed=7).process(dataset).to_list() == expected


@pytest.mark.parametrize("config", [{}, {"select_num": 100}, {"select_ratio": 0.0}])
def test_seed_keeps_noop_and_empty_selection_behavior(dataset, config):
    assert (
        RandomSelector(seed=7, **config).process(dataset).to_list()
        == RandomSelector(**config).process(dataset).to_list()
    )


def test_does_not_change_numpy_global_rng(dataset):
    before = np.random.get_state()
    RandomSelector(select_num=17, seed=7).process(dataset)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
