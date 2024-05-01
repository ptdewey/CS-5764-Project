#!/usr/bin/env bash

# create temporary rscript
tempscript=$(mktemp /tmp/knitscript.XXXXX) || { echo "Failed to create temp script"; return 1 2>/dev/null || exit 1; }

echo "library(rmarkdown); rmarkdown::render('src/index.Rmd', output_dir='docs'); rmarkdown::render('src/mnist_exploratory.Rmd', output_dir='docs')" >> $tempscript

cat $tempscript
if ! Rscript $tempscript; then
    echo "Failed to run Rscript"
    rm -f $tempscript
    return 1 2>/dev/null || exit 1
fi

# Clean up temporary script
rm -f $tempscript
