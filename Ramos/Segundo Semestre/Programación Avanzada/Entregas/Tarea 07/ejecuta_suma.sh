#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Uso: !/content/ejecutar_suma.sh <a> <b>"
    exit 1
fi

python3 suma.py "$1" "$2"
