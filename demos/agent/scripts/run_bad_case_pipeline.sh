#!/usr/bin/env bash
# 一键：bad case 冒烟 / 全量菜谱 / 仅后处理 / 单元测试
# 在仓库根目录执行，或从任意目录：bash demos/agent/scripts/run_bad_case_pipeline.sh <cmd>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$ROOT"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
  else
    PY="python3"
  fi
fi
SMOKE_OUT="${SMOKE_OUT:-./outputs/agent_bad_case_smoke/processed.jsonl}"
FULL_OUT="${FULL_OUT:-./outputs/agent_quality/processed.jsonl}"
CAL_OUT="${CAL_OUT:-./outputs/agent_quality/bad_case_calibration.json}"
RECIPE_FULL="demos/agent/agent_interaction_quality_analysis.yaml"
RECIPE_SMOKE="demos/agent/minimal_configs/09_bad_case_smoke.yaml"

usage() {
  cat <<'EOF'
用法（在仓库根目录，或任意目录调用本脚本）:

  bash demos/agent/scripts/run_bad_case_pipeline.sh smoke
      dj-process 最小 09 配置 → 校验 → 分位数 → cohort → 切片示例 → HTML 报告（可 SKIP_AUTO_REPORT=1）

  bash demos/agent/scripts/run_bad_case_pipeline.sh report [选项] [JSONL...] [OUT.html]
      一键：校验 + 生成自助 HTML 报告（图表+表；进阶说明折叠在页内）
      可写多个 JSONL（分批导出、无需先 cat 合并）；若最后一项路径以 .html/.htm 结尾则视为输出，其余均为输入。
      无位置参数时默认输入: ./outputs/agent_quality/processed.jsonl
      默认开启：--bilingual、--llm-summary（无 API Key 时导读会降级为规则摘要并打 WARNING）
      选项：--no-bilingual | --no-llm-summary（显式关闭）

  bash demos/agent/scripts/run_bad_case_pipeline.sh postprocess [JSONL]
      对已导出的 jsonl 跑 verify + percentiles + cohorts + slice（深度调试用；不生成 HTML）

  bash demos/agent/scripts/run_bad_case_pipeline.sh calibrate-and-slice [JSONL]
      从 JSONL 写 P95 校准文件到 CAL_OUT，并打印下一步 YAML 配置提示

  bash demos/agent/scripts/run_bad_case_pipeline.sh full
      跑完整 agent_interaction_quality_analysis.yaml（需 LLM API；耗时/费用高）

  bash demos/agent/scripts/run_bad_case_pipeline.sh unittest
      仅跑 agent_bad_case_signal_mapper 单元测试（无需 dj-process）

环境变量:
  PYTHON   python 可执行文件（默认 python3）
  SMOKE_OUT / FULL_OUT / CAL_OUT  输出路径覆盖
  SKIP_AUTO_REPORT=1  smoke/full 末尾不自动生成 HTML
  BAD_CASE_REPORT_LLM=0  report / smoke / full 末尾生成 HTML 时不加 --llm-summary（默认会加）
  BAD_CASE_REPORT_BILINGUAL=0  同上，不加 --bilingual（默认会加）
  BAD_CASE_REPORT_LLM_TIMEOUT_SEC  导读 HTTP 读超时秒数（默认 120；亦可直接传 generate_bad_case_report.py --llm-timeout-sec）

日常只看结论: demos/agent/BAD_CASE_REPORT.md
详见: demos/agent/BAD_CASE_INSIGHTS.md 与 demos/agent/scripts/README.md
EOF
}

run_report() {
  local use_bilingual="${1:-1}"
  local use_llm="${2:-1}"
  shift 2
  local -a rest=("$@")
  local -a inputs=()
  local out=""
  if [[ ${#rest[@]} -eq 0 ]]; then
    inputs=("$FULL_OUT")
  else
    local last="${rest[-1]}"
    if [[ "$last" == *.html ]] || [[ "$last" == *.htm ]]; then
      out="$last"
      if [[ ${#rest[@]} -gt 1 ]]; then
        inputs=("${rest[@]:0:${#rest[@]}-1}")
      else
        inputs=("$FULL_OUT")
      fi
    else
      inputs=("${rest[@]}")
    fi
  fi
  for f in "${inputs[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: input not found: $f" >&2
      exit 3
    fi
  done
  if [[ -z "$out" ]]; then
    out="${inputs[0]%.jsonl}_bad_case_report.html"
  fi
  echo "==> verify_bad_case_export.py (${#inputs[@]} file(s))"
  local -a _ver=(demos/agent/scripts/verify_bad_case_export.py)
  for f in "${inputs[@]}"; do
    _ver+=(--input "$f")
  done
  "$PY" "${_ver[@]}"
  echo "==> generate_bad_case_report.py -> $out"
  local -a _rep_args
  _rep_args=(--output "$out")
  for f in "${inputs[@]}"; do
    _rep_args+=(--input "$f")
  done
  if [[ "$use_bilingual" == "1" ]]; then
    _rep_args+=(--bilingual)
  fi
  if [[ "$use_llm" == "1" ]]; then
    _rep_args+=(--llm-summary)
  fi
  "$PY" demos/agent/scripts/generate_bad_case_report.py "${_rep_args[@]}"
  echo "Open in browser (PII-safe): $out"
  echo "Open PII audit copy: ${out%.html}_pii_audit.html"
}

run_postprocess() {
  local input="${1:-$SMOKE_OUT}"
  if [[ ! -f "$input" ]]; then
    echo "ERROR: input not found: $input" >&2
    echo "先运行: bash demos/agent/scripts/run_bad_case_pipeline.sh smoke" >&2
    exit 3
  fi
  echo "==> verify_bad_case_export.py"
  "$PY" demos/agent/scripts/verify_bad_case_export.py --input "$input"
  echo "==> compute_percentile_thresholds.py (console report)"
  "$PY" demos/agent/scripts/compute_percentile_thresholds.py --input "$input"
  local cohort_csv="${input%.jsonl}_cohort_summary.csv"
  echo "==> analyze_bad_case_cohorts.py -> $cohort_csv"
  "$PY" demos/agent/scripts/analyze_bad_case_cohorts.py --input "$input" --out-csv "$cohort_csv"
  local slice_out="${input%.jsonl}_high_precision_sample.jsonl"
  echo "==> slice_export_by_tier.py (high_precision, limit 20) -> $slice_out"
  "$PY" demos/agent/scripts/slice_export_by_tier.py \
    --input "$input" --tier high_precision --output "$slice_out" --limit 20
  echo "Done postprocess on $input"
}

cmd="${1:-}"
case "$cmd" in
  smoke)
    if ! command -v dj-process >/dev/null 2>&1; then
      echo "ERROR: dj-process not in PATH. 请先激活本仓库环境，例如:" >&2
      echo "  uv venv && source .venv/bin/activate && uv pip install -e ." >&2
      echo "  见 demos/agent/minimal_configs/README.md" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$SMOKE_OUT")"
    echo "==> dj-process (minimal bad-case smoke, no LLM API)"
    dj-process --config "$RECIPE_SMOKE"
    run_postprocess "$SMOKE_OUT"
    if [[ "${SKIP_AUTO_REPORT:-}" != "1" ]]; then
      run_report "${BAD_CASE_REPORT_BILINGUAL:-1}" "${BAD_CASE_REPORT_LLM:-1}" \
        "$SMOKE_OUT" "${SMOKE_OUT%.jsonl}_bad_case_report.html"
    fi
    ;;
  report)
    shift
    _rb="${BAD_CASE_REPORT_BILINGUAL:-1}"
    _rl="${BAD_CASE_REPORT_LLM:-1}"
    _rpos=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --bilingual) _rb=1; shift ;;
        --no-bilingual) _rb=0; shift ;;
        --llm-summary) _rl=1; shift ;;
        --no-llm-summary) _rl=0; shift ;;
        *) _rpos+=("$1"); shift ;;
      esac
    done
    run_report "$_rb" "$_rl" "${_rpos[@]}"
    ;;
  postprocess)
    run_postprocess "${2:-$SMOKE_OUT}"
    ;;
  calibrate-and-slice)
    input="${2:-$FULL_OUT}"
    if [[ ! -f "$input" ]]; then
      echo "ERROR: $input not found. Set path or run full/smoke first." >&2
      exit 3
    fi
    mkdir -p "$(dirname "$CAL_OUT")"
    echo "==> write calibration JSON -> $CAL_OUT"
    "$PY" demos/agent/scripts/compute_percentile_thresholds.py \
      --input "$input" \
      --write-calibration "$CAL_OUT" \
      --calibration-percentile 95
    cat <<EOF

下一步：在 agent_bad_case_signal_mapper 中增加:
  auto_calibrate_thresholds: true
  calibration_json_path: "$CAL_OUT"

然后对**同结构**数据再跑一遍 dj-process（或全量重跑）。
EOF
    ;;
  full)
    if ! command -v dj-process >/dev/null 2>&1; then
      echo "ERROR: dj-process not in PATH. 见 smoke 命令的说明。" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$FULL_OUT")"
    echo "WARN: full recipe calls many LLM APIs (cost + keys). export_path -> $FULL_OUT"
    dj-process --config "$RECIPE_FULL"
    echo "==> verify (with agent_insight_llm if pipeline completed step 10)"
    "$PY" demos/agent/scripts/verify_bad_case_export.py --input "$FULL_OUT" --require-insight || \
      "$PY" demos/agent/scripts/verify_bad_case_export.py --input "$FULL_OUT"
    run_postprocess "$FULL_OUT"
    if [[ "${SKIP_AUTO_REPORT:-}" != "1" ]]; then
      run_report "${BAD_CASE_REPORT_BILINGUAL:-1}" "${BAD_CASE_REPORT_LLM:-1}" \
        "$FULL_OUT" "${FULL_OUT%.jsonl}_bad_case_report.html"
    fi
    ;;
  unittest)
    echo "==> unittest agent_bad_case_signal_mapper"
    PYTHONPATH=. "$PY" -m unittest tests.ops.mapper.test_agent_bad_case_signal_mapper -v
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 2
    ;;
esac
