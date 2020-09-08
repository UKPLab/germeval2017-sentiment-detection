#!/usr/bin/env bash

# Backup embedings in a gziped file
# See https://www.unixmen.com/performing-incremental-backups-using-tar/
tar --listed-incremental=backup.snapshot -czvf embeddings.tar.gz ./**/*
