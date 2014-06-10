#!/usr/bin/env bash

while IFS= read -rd '' file; do
  dirname="${file%/*}"
  basename="${file##*/}"
  mv --backup=numbered -- "${file}" "${dirname}/${basename//\\/_}"
done < <(find /full/path/to/dir -depth -name '*\\*' -print0)
