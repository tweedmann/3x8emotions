3x8emotions
================
Tobias Widmann & Maximilian Wich
10/7/2021

Repo containing code and models for 3 different tools to measure appeals
to 8 discrete emotions in German political text, as described and
validated in the following article:

Please start by reading this article which contains information about
the creation and performance of the different tools. These tools are
free to use for academic research. In case you use one or multiple of
these, please always cite the article above.

In order to obtain all necessary files, start by downloading this repo
as a .zip file. The folder contains all scripts to apply the (1) ed8
dictionary, (2) the neural network models based on locally trained word
embeddings, and (3) the ELECTRA model.

## ed8

The `ed8 dictionary` is provided in YAML format and can be applied via
the the `quanteda` package. The dictionary and the R script to apply the
dictionary to a data frame with a ‘text’ column can be found in the
folder `"./ed8"`.

## Neural Network Classifiers

The neural network classifiers and word embedding model are provided in
the folder `"./neuralnet"`. The code for turning text into numerical
vectors and subsequently applying the neural network classifiers can be
found in the R script —. The machine learning models are provided in the
folder `"./neuralnet/models"`.

## ELECTRA Model

You can also embed plots, for example:
