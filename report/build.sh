#!/usr/bin/env bash
# Build every LaTeX document under report/ with Tectonic, then normalise each PDF to
# PDF-1.4 with Ghostscript so GitHub's inline viewer renders it.
#
# Usage:
#   report/build.sh            # build all *.tex under report/
#   report/build.sh <path>     # build only *.tex under <path>
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
search="${1:-$here}"

command -v tectonic >/dev/null || { echo "error: tectonic not found on PATH" >&2; exit 1; }
command -v gs >/dev/null       || { echo "error: ghostscript (gs) not found on PATH" >&2; exit 1; }

count=0
while IFS= read -r -d '' tex; do
    # only compile top-level documents; skip \input fragments (e.g. dissertation chapters/)
    grep -q 'documentclass' "$tex" || { echo "-- skip fragment $tex"; continue; }
    echo "==> building $tex"
    dir="$(dirname "$tex")"
    tectonic --synctex --keep-logs --keep-intermediates -o "$dir" "$tex"
    pdf="${tex%.tex}.pdf"
    if [[ -f "$pdf" ]]; then
        tmp="${pdf%.pdf}.gs.pdf"
        gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 \
           -dPDFSETTINGS=/prepress -sOutputFile="$tmp" "$pdf"
        mv "$tmp" "$pdf"
    fi
    count=$((count + 1))
done < <(find "$search" -name '*.tex' -print0)

echo "Done — rebuilt $count document(s)."
