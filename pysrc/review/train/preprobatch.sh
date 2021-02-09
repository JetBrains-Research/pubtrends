#!/usr/bin/env bash

python -W ignore -m preprocess \
    -mode preprocess \
    -root2data pubmed \
    -split dev \
    -save_prefix pubmedtop50 \
    -n_procs 5

python -W ignore -m preprocess \
    -mode preprocess \
    -root2data pubmed \
    -split train \
    -save_prefix pubmedtop50 \
    -n_procs 8

python -W ignore -m preprocess \
    -mode merge \
    -root2data pubmed \
    -split dev \
    -save_prefix pubmedtop50

python -W ignore -m preprocess \
    -mode merge \
    -root2data pubmed \
    -split train \
    -save_prefix pubmedtop50
