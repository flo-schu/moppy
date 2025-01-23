#!/bin/bash

# Input string
input_string="$1"

# Regular expression to match the required pattern
regex="^bump version [0-9]+\.[0-9]+\.[0-9]+[a-zA-Z]?[0-9]* -> [0-9]+\.[0-9]+\.[0-9]+[a-zA-Z]?[0-9]*$"

if [[ $input_string =~ $regex ]]; then
  echo "The string matches the required pattern."
else
  echo "The string does not match the required pattern."
fi
