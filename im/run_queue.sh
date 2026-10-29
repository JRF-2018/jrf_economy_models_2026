#!/bin/bash
# 重い実験を順に、並列 4 で回す(WSL が落ちないように、メモリの上限つき)
cd "$(dirname "$0")"
ulimit -v 4000000
export IM_WORKERS=4
for s in "$@"; do
  python3 -W ignore results_$s.py > results_$s.txt 2> results_$s.err
  echo "$s done $(date +%T)"
done
