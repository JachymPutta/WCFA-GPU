#!/bin/bash
for D in tests/* ; do
    echo "Executing: $D";
    ./wcfa "${D#tests/}";
    echo "$D done.";
done

