#!/usr/bin/env bash

# $HOME /home/umar.salman

# Installing Libraries
pip install requirements.txt

# Make data directory
DATA_DIR=${PWD}/data
mkdir $DATA_DIR


# Download Squad dataset
SQUAD_DIR=$DATA_DIR/Squad
mkdir $SQUAD_DIR
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json


# Download GloVe
GLOVE_DIR=$DATA_DIR/Glove
mkdir $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR

# Download Spacy
# python3 -m spacy download en_core_web_sm