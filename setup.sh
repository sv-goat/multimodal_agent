#!/usr/bin/env bash
set -e

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
