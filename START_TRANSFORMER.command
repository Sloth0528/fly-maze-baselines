#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Please install dependencies using README.md first.'
  exit 1
fi
exec .venv/bin/python run_transformer.py "$@"
