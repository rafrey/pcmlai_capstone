#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
venv_dir="$repo_root/.venv"
python_bin="$venv_dir/bin/python"
fix_script="$repo_root/scripts/fix_tf_cuda_venv.sh"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    echo "Install it first, then rerun this script." >&2
    exit 1
fi

echo "Creating or refreshing virtual environment at $venv_dir"
uv venv --allow-existing "$venv_dir"

if [[ ! -x "$python_bin" ]]; then
    echo "Expected Python interpreter was not created at $python_bin." >&2
    exit 1
fi

echo "Installing project dependencies with uv"
uv pip install --python "$python_bin" -r "$repo_root/requirements.txt"

echo "Checking whether TensorFlow can load GPU libraries cleanly"
probe_output_file="$(mktemp)"
trap 'rm -f "$probe_output_file"' EXIT

probe_command='import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
probe_exit_code=0

if ! "$python_bin" -c "$probe_command" >"$probe_output_file" 2>&1; then
    probe_exit_code=$?
fi

probe_output="$(cat "$probe_output_file")"

if [[ "$probe_exit_code" -eq 0 ]] && [[ ! "$probe_output" =~ (Cannot\ dlopen\ some\ GPU\ libraries|cannot\ open\ shared\ object\ file|No\ such\ file\ or\ directory|libcuda|libcud|libcublas|libcusolver|libcusparse|libcufft|libcurand|libnccl|libnvrtc|libnvJitLink|libcupti|libtensorflow_framework) ]]; then
    echo "$probe_output"
    echo "TensorFlow probe completed without missing-library warnings. No CUDA library repair was needed."
    exit 0
fi

echo "$probe_output" >&2

if [[ "$(uname -s)" == "Linux" ]] && [[ "$probe_output" =~ (Cannot\ dlopen\ some\ GPU\ libraries|cannot\ open\ shared\ object\ file|No\ such\ file\ or\ directory|libcuda|libcud|libcublas|libcusolver|libcusparse|libcufft|libcurand|libnccl|libnvrtc|libnvJitLink|libcupti|libtensorflow_framework) ]]; then
    echo "TensorFlow probe indicates missing shared libraries. Running CUDA fixup script."
    "$fix_script"

    echo "Re-checking TensorFlow GPU library loading after CUDA fixup"
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    if python -c "$probe_command"; then
        echo "TensorFlow probe succeeded after CUDA fixup."
        exit 0
    fi

    echo "TensorFlow still reported GPU library loading issues after CUDA fixup." >&2
    exit 1
fi

echo "TensorFlow probe failed for a reason other than missing shared libraries. Skipping CUDA fixup." >&2
exit 1