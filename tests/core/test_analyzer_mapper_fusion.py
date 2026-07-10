from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from data_juicer.core import Analyzer


def test_fused_tagging_mapper_is_analyzed():
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.cfg = SimpleNamespace(
        np=1,
        auto=False,
        process=[],
        op_fusion=True,
        fusion_strategy="greedy",
        open_monitor=False,
        use_cache=False,
        percentiles=[],
        save_stats_in_one_file=False,
    )
    analyzer.work_dir = "test-work-dir"
    analyzer.analysis_path = "test-analysis-path"
    analyzer.exporter = MagicMock()
    fused_op = SimpleNamespace(
        _name="fused:test_tagging_a,test_tagging_b",
        _contains_tagging_ops=True,
    )
    dataset = MagicMock()
    dataset.process.return_value = dataset
    fuse_operators = MagicMock(return_value=[fused_op])

    with (
        patch("data_juicer.core.analyzer.load_ops", return_value=[fused_op]),
        patch("data_juicer.core.analyzer.fuse_operators", fuse_operators),
        patch("data_juicer.core.analyzer.OverallAnalysis"),
        patch("data_juicer.core.analyzer.ColumnWiseAnalysis"),
        patch("data_juicer.core.analyzer.CorrelationAnalysis"),
    ):
        result = analyzer.run(dataset=dataset)

    assert result is dataset
    assert fuse_operators.call_args.kwargs["mapper_fusion"] is False
    dataset.process.assert_called_once_with(
        fused_op,
        work_dir="test-work-dir",
        open_monitor=False,
    )
