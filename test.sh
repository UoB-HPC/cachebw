#!/bin/bash

for i in {0..14}
do
  ./cachebw $((2**i)) 1000
done
