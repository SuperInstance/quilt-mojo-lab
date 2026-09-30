#!/bin/sh
# Build the C quilt kernel as a shared object for the ctypes bridge.
set -e
cd "$(dirname "$0")"
gcc -O3 -fPIC -shared -o libflatquilt.so flat_quilt.c -lm
gcc -O3 -fPIC -shared -o libsoaquilt.so soa_quilt.c -lm
echo "built c/libflatquilt.so c/libsoaquilt.so"
