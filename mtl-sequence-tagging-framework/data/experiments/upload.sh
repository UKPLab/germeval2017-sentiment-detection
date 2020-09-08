#!/usr/bin/env bash

if [ -z $1 ]; then
  echo "Please provide a hostname or an IP address as the upload target. Example: ./upload.sh example.com /data/store"
  exit
fi

if [ -z $2 ]; then
  echo "Please provide a path on the upload target. Example: ./upload.sh example.com /data/store"
  exit
fi

HOST=$1
HOST_PATH=$2
DATE=`date +%Y-%m-%d`
NAME=experiment_data.tar.gz

# gzip this directory
tar -czvf ${NAME} ./**/data/*

MD5_SUM=$(md5sum ${NAME} | awk '{print $1}')

FULL_NAME=$(printf "%s_%s_%s" "$DATE" "$MD5_SUM" "$NAME")

mv ${NAME} ${FULL_NAME}

echo scp ${FULL_NAME} ${HOST}:${HOST_PATH}
scp ${FULL_NAME} ${HOST}:${HOST_PATH}

rm -f ${FULL_NAME}
