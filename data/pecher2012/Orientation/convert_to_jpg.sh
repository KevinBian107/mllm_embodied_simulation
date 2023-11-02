#!/bin/bash

for file in ./*; do
    base_name=$(basename "$file")
    new_name="${base_name%.*}.pict"
    cp "$file" "$new_name"
    convert "$new_name" "${new_name%.*}.jpg"
done