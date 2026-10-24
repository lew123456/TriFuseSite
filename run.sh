#!/usr/bin/env bash
set -euo pipefail
task="${1:?Specify ppi, smpi, dpi, or rpi}"
shift
case "$task" in ppi|smpi|dpi|rpi) ;; *) echo 'Unknown task' >&2; exit 2 ;; esac
python main_train.py --config "config/${task}.yaml" "$@"
