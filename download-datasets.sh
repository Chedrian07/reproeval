#!/usr/bin/env bash
# Download all benchmark datasets for codebench
#
# Usage:
#   ./download-datasets.sh                          # download all
#   ./download-datasets.sh humaneval_plus mbpp_plus  # specific ones
#   ./download-datasets.sh --list                    # show available
#   ./download-datasets.sh --force                   # re-download even if exists

set -euo pipefail
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════╗"
echo "║   codebench — Dataset Downloader         ║"
echo "╚══════════════════════════════════════════╝"
echo

# Check for datasets library
if ! python3 -c "import datasets" 2>/dev/null; then
    echo "Installing 'datasets' library (HuggingFace)..."
    pip install datasets -q
    echo
fi

# Run the download script
python3 scripts/download_datasets.py "$@"
