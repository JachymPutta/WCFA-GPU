#!/bin/bash
while read f; do
    echo "Verifying stores:"
    ./compare_stores "res/$f" "scheme/$f"
    echo "Done."
done <tests.txt
