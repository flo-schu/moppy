#!/bin/bash

# Directory to scan (current directory by default)
dir="$1"
branch="$2"

# Loop through all files in the directory (recursively, if desired)
for file in "$dir"/*; do
  # Check if it's a regular file
  if [[ -f "$file" ]]; then
    # Search for the string "TEST:OK" in the file
    if grep -q "TEST:OK" "$file"; then
      echo "File '$file' contains 'TEST:OK'"
    else
      echo "File '$file' does not contain 'TEST:OK'"
      exit 1
    fi

    # Check if the filename contains the specified word
    if [[ "$file" == "$dir/${branch}_"*".txt" ]]; then
      echo "Test belongs to '$branch' branch"
      echo "$dir/${branch}_*.txt"
    else
      echo "Test does not belong to '$branch' branch."
      exit 1
    fi
  fi
done