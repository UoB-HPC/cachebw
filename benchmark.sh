#!/bin/bash

set -eu

usage () {
    echo "Usage: $0 [-h] [-n MAXN] [-r REPS]"
}

# Default options
n=11
reps=10000

while getopts ":n:r:h" opt; do
    case "$opt" in
        n)
            n=$OPTARG
            ;;
        r)
            reps=$OPTARG
            ;;
        h)
            usage
            exit
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage >&2
            exit 1
            ;;
        :)
            echo "Invalid option: -$OPTARG requires an argument" >&2
            usage >&2
            exit 2
            ;;
    esac
done
shift $(( OPTIND -1 ))


# Benchmark
for i in $(seq 0 "$n"); do
    ./cachebw $((2**i)) "$reps"
done
