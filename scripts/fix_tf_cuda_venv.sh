#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This script only supports Linux or WSL2 environments." >&2
    exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    venv_dir="$VIRTUAL_ENV"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    venv_dir="$repo_root/.venv"
else
    echo "No virtual environment detected. Activate one first or create $repo_root/.venv." >&2
    exit 1
fi

python_bin="$venv_dir/bin/python"

if [[ ! -x "$python_bin" ]]; then
    echo "Python executable not found at $python_bin." >&2
    exit 1
fi

mapfile -t metadata < <("$python_bin" - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import sys

tf_spec = find_spec("tensorflow")
if tf_spec is None or not tf_spec.submodule_search_locations:
    print("ERROR=TensorFlow is not installed in the selected virtual environment.")
    raise SystemExit(1)

tf_dir = Path(list(tf_spec.submodule_search_locations)[0]).resolve()
nvidia_dir = tf_dir.parent / "nvidia"

if not nvidia_dir.is_dir():
    print("ERROR=The NVIDIA wheel packages were not found. Install tensorflow[and-cuda] first.")
    raise SystemExit(1)

lib_dirs = sorted(
    path.resolve() for path in nvidia_dir.glob("*/lib") if path.is_dir() and any(path.glob("*.so*"))
)

if not lib_dirs:
    print("ERROR=No NVIDIA shared library directories were found under site-packages/nvidia.")
    raise SystemExit(1)

ptxas_candidates = sorted(path.resolve() for path in nvidia_dir.glob("*/bin/ptxas") if path.is_file())

print(f"TF_DIR={tf_dir}")
print(f"NVIDIA_DIR={nvidia_dir.resolve()}")
print(f"LIB_COUNT={len(lib_dirs)}")
print("LIB_DIRS=" + ":".join(str(path) for path in lib_dirs))
if ptxas_candidates:
    print(f"PTXAS={ptxas_candidates[0]}")
PY
)

declare -A info=()

for line in "${metadata[@]}"; do
    if [[ "$line" == ERROR=* ]]; then
        echo "${line#ERROR=}" >&2
        exit 1
    fi

    key="${line%%=*}"
    value="${line#*=}"
    info["$key"]="$value"
done

tf_dir="${info[TF_DIR]:-}"
nvidia_dir="${info[NVIDIA_DIR]:-}"
lib_count="${info[LIB_COUNT]:-0}"
lib_dirs="${info[LIB_DIRS]:-}"
ptxas_path="${info[PTXAS]:-}"

if [[ -z "$tf_dir" || -z "$nvidia_dir" ]]; then
    echo "Failed to resolve TensorFlow and NVIDIA package locations." >&2
    exit 1
fi

if [[ ! -d "$tf_dir" || ! -d "$nvidia_dir" ]]; then
    echo "Resolved package directories do not exist." >&2
    exit 1
fi

if [[ -z "$lib_dirs" ]]; then
    echo "No NVIDIA library directories were resolved." >&2
    exit 1
fi

echo "Using virtual environment: $venv_dir"
echo "TensorFlow package directory: $tf_dir"
echo "NVIDIA package directory: $nvidia_dir"
echo "Linking shared libraries into TensorFlow package directory..."

pushd "$tf_dir" >/dev/null
shopt -s nullglob
libs=(../nvidia/*/lib/*.so*)

if (( ${#libs[@]} == 0 )); then
    echo "No shared libraries were found to link." >&2
    popd >/dev/null
    exit 1
fi

for lib in "${libs[@]}"; do
    ln -sfn "$lib" .
done

shopt -u nullglob
popd >/dev/null

echo "Linked ${#libs[@]} shared library files from $lib_count NVIDIA library directories."

if [[ -n "$ptxas_path" ]]; then
    ln -sfn "$ptxas_path" "$venv_dir/bin/ptxas"
    echo "Linked ptxas into $venv_dir/bin/ptxas"
else
    echo "ptxas was not found in the installed NVIDIA packages. Skipping that link."
fi

cuda_env_file="$venv_dir/lib/tensorflow-cuda-paths.sh"
{
    printf '%s\n' 'if [ -n "${LD_LIBRARY_PATH:-}" ]; then'
    printf '    export LD_LIBRARY_PATH="%s:${LD_LIBRARY_PATH}"\n' "$lib_dirs"
    printf '%s\n' 'else'
    printf '    export LD_LIBRARY_PATH="%s"\n' "$lib_dirs"
    printf '%s\n' 'fi'
} >"$cuda_env_file"

activate_file="$venv_dir/bin/activate"
"$python_bin" - <<'PY' "$activate_file" "$cuda_env_file"
from pathlib import Path
import sys

activate_path = Path(sys.argv[1])
cuda_env_path = Path(sys.argv[2])
start_marker = "# >>> tensorflow cuda paths >>>"
end_marker = "# <<< tensorflow cuda paths <<<"
block = (
    f"{start_marker}\n"
    f'if [ -f "{cuda_env_path}" ]; then\n'
    f'    . "{cuda_env_path}"\n'
    f"fi\n"
    f"{end_marker}\n"
)

content = activate_path.read_text()
if start_marker in content and end_marker in content:
    before, _, remainder = content.partition(start_marker)
    _, _, after = remainder.partition(end_marker)
    if after.startswith("\n"):
        after = after[1:]
    content = before.rstrip("\n") + "\n\n" + block + after.lstrip("\n")
else:
    content = content.rstrip("\n") + "\n\n" + block

activate_path.write_text(content)
PY

echo "Wrote CUDA loader paths to $cuda_env_file"
echo "Updated $activate_file to source the CUDA loader paths on activation"

echo "TensorFlow CUDA venv fixup complete."