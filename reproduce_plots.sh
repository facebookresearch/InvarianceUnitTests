# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#!/bin/bash

# Untar results.tar.gz
if [ ! -d "results_processed" ]; then
    tar -xvzf results_processed.tar.gz
    echo "unzip"
fi

## Plot figures 1, 2 ##
echo "Plot figures 1 & 2"
python3.8 plot_results.py -dirname results/ -commit e717c2ff36 --load
