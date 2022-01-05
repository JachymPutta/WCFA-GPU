#!/bin/bash
for D in tests/* ; do
    echo "Executing: $D";
    ./main "${D#tests/}";
    echo "$D done.";
done

