#!/bin/bash

# Pipeline Performance Test Runner
# This script runs the pipeline performance test with different configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RECIPE_PATH=""
DATASET_PATH=""
OUTPUT_DIR="./outputs/pipeline_perf_test"
USE_BENCHMARK_FRAMEWORK=false
VERBOSE=false

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -r, --recipe-path PATH     Path to the recipe YAML file (required)"
    echo "  -d, --dataset-path PATH    Path to the dataset file (required)"
    echo "  -o, --output-dir PATH      Output directory (default: ./outputs/pipeline_perf_test)"
    echo "  -b, --benchmark-framework  Use comprehensive benchmark framework"
    echo "  -v, --verbose              Enable verbose logging"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -r configs/data_juicer_recipes/alpaca-cot-en-refine.yaml -d data/sample.jsonl"
    echo "  $0 --recipe-path configs/demo/analyzer.yaml --dataset-path demos/data/demo-dataset.jsonl --verbose"
    echo "  $0 -r configs/demo/analyzer.yaml -d demos/data/demo-dataset.jsonl -b"
}

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--recipe-path)
            RECIPE_PATH="$2"
            shift 2
            ;;
        -d|--dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--benchmark-framework)
            USE_BENCHMARK_FRAMEWORK=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$RECIPE_PATH" ]]; then
    print_error "Recipe path is required"
    print_usage
    exit 1
fi

if [[ -z "$DATASET_PATH" ]]; then
    print_error "Dataset path is required"
    print_usage
    exit 1
fi

# Check if files exist
if [[ ! -f "$RECIPE_PATH" ]]; then
    print_error "Recipe file not found: $RECIPE_PATH"
    exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
    print_error "Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

print_info "Starting Pipeline Performance Test"
print_info "Project root: $PROJECT_ROOT"
print_info "Recipe path: $RECIPE_PATH"
print_info "Dataset path: $DATASET_PATH"
print_info "Output directory: $OUTPUT_DIR"
print_info "Use benchmark framework: $USE_BENCHMARK_FRAMEWORK"
print_info "Verbose: $VERBOSE"

# Build command
CMD="python tests/benchmark_performance/perf_test_pipeline_comparison.py"
CMD="$CMD --recipe-path \"$RECIPE_PATH\""
CMD="$CMD --dataset-path \"$DATASET_PATH\""
CMD="$CMD --output-dir \"$OUTPUT_DIR\""

if [[ "$USE_BENCHMARK_FRAMEWORK" == true ]]; then
    CMD="$CMD --use-benchmark-framework"
fi

if [[ "$VERBOSE" == true ]]; then
    CMD="$CMD --verbose"
fi

print_info "Running command: $CMD"

# Run the test
if eval $CMD; then
    print_success "Pipeline performance test completed successfully!"

    # Show results summary
    RESULTS_FILE="$OUTPUT_DIR/results.json"
    REPORT_FILE="$OUTPUT_DIR/performance_report.md"

    if [[ -f "$RESULTS_FILE" ]]; then
        print_info "Results saved to: $RESULTS_FILE"
    fi

    if [[ -f "$REPORT_FILE" ]]; then
        print_info "Report generated: $REPORT_FILE"
        echo ""
        print_info "Report preview:"
        echo "=================="
        head -20 "$REPORT_FILE"
        echo "..."
        echo "=================="
    fi

else
    print_error "Pipeline performance test failed!"
    exit 1
fi
