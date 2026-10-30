#!/usr/bin/env bash

# Install the native Graphviz tools inside a virtual environment on Debian/Ubuntu.
# APT resolves the missing packages; extraction requires no root privileges.
set -euo pipefail

if command -v dot >/dev/null 2>&1; then
    dot -V
    exit 0
fi

for required_command in apt-get dpkg-deb awk; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "This installer requires Debian/Ubuntu tools; install Graphviz with your system package manager." >&2
        exit 1
    fi
done

graphviz_venv=$(realpath "${1:-.venv}")
if [[ ! -x "$graphviz_venv/bin/python" ]]; then
    echo "Create the virtual environment before installing Graphviz: $graphviz_venv" >&2
    exit 1
fi
if [[ -x "$graphviz_venv/bin/dot" ]] && "$graphviz_venv/bin/dot" -V; then
    exit 0
fi

graphviz_downloads=$(mktemp -d "${TMPDIR:-/tmp}/larch-graphviz.XXXXXX")
cleanup() { rm -rf -- "$graphviz_downloads"; }
trap cleanup EXIT

apt-get --simulate install --no-install-recommends graphviz > "$graphviz_downloads/plan.txt"
mapfile -t graphviz_packages < <(
    # Upgrades include an [installed-version] field before (candidate-version).
    awk '/^Inst / {
        package = $2
        sub(/^[^(]*\(/, "")
        print package "=" $1
    }' "$graphviz_downloads/plan.txt"
)
if [[ ${#graphviz_packages[@]} == 0 ]]; then
    echo "Graphviz is already installed; ensure its dot executable is on PATH." >&2
    exit 1
fi

graphviz_root="$graphviz_venv/share/graphviz"
mkdir -p "$graphviz_root"
(
    cd "$graphviz_downloads"
    apt-get download "${graphviz_packages[@]}"
    for graphviz_package in ./*.deb; do
        dpkg-deb -x "$graphviz_package" "$graphviz_root"
    done
)

cat > "$graphviz_venv/bin/dot" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
graphviz_venv=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
graphviz_root="$graphviz_venv/share/graphviz"
graphviz_libraries=()
for graphviz_directory in "$graphviz_root"/usr/lib/*-linux-gnu "$graphviz_root/usr/lib"; do
    if [[ -d "$graphviz_directory" ]]; then
        graphviz_libraries+=("$graphviz_directory")
        if [[ -d "$graphviz_directory/graphviz" ]]; then
            export GVBINDIR="$graphviz_directory/graphviz"
        fi
    fi
done
printf -v graphviz_library_path '%s:' "${graphviz_libraries[@]}"
export LD_LIBRARY_PATH="${graphviz_library_path%:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$graphviz_root/usr/bin/dot" "$@"
WRAPPER
chmod +x "$graphviz_venv/bin/dot"
"$graphviz_venv/bin/dot" -c
"$graphviz_venv/bin/dot" -V
