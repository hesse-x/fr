#!/bin/bash
path=$(dirname "$(realpath "$0")")
PYTHONPATH=$path python3 -m pytest test/unit/mesh_test.py
