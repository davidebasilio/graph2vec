#!/bin/bash

basedir=$(dirname "$0")

datadirs=$(find $basedir/data/* -type d)

mkdir -p $basedir/{embeddings,plots}

for d in $datadirs; do
  graph=${d##*/}

  echo Analyzing $graph.
  input_dir=$basedir/data/$graph
  embeddings_file=$basedir/embeddings/$graph
  python3 $basedir/src/graph2vec.py --input-path $input_dir --output-path $embeddings_file --dimensions 256 --workers 32

  echo Plotting $graph.
  python3 $basedir/plot.py $embeddings_file
done
