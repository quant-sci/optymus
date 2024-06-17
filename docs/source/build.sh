#!/bin/bash
cp -r notebooks docs/source/examples/

PYTHONPATH=./src sphinx-build -b html docs/source docs/build