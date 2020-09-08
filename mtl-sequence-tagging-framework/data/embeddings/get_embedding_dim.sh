#!/usr/bin/env bash

embedding_file=$1
num_cols=`awk '{print NF}' $embedding_file | sort -nu | tail -n 1`

echo $((num_cols - 1))
