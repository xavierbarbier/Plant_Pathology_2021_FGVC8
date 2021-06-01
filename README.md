# Plant Pathology 2021 FGVC8


## Intro

Plant Pathology 2021 - FGVC8 is a Kaggle competition launched on march 15 2021 and closed on mai 27 2021 : https://www.kaggle.com/c/plant-pathology-2021-fgvc8

The main objective of the competition is to develop machine learning-based models to accurately classify a given leaf image from the test dataset to a particular disease category, and to identify an individual disease from multiple disease symptoms on a single leaf image.

![Plant Pathology 2021 FGVC8](https://github.com/xavierbarbier/Plant_Pathology_2021_FGVC8/blob/main/plant_pathology.png)

## Configuration
Environment : Kaggle notebook with TPU

Libraries used : TensorFlow, Keras, Pandas, Numpy, Matplotlib

## EDA

Number of images: 18632

We have 12 labels combinaisons made out of 6 labels. In the training phase, our choice will be to use a multilabels classification approach. Therefore to predict each label independently.

Notebook => https://github.com/xavierbarbier/Plant_Pathology_2021_FGVC8/blob/main/plant-pathology-2021-fgvc8-eda.ipynb

## Training

The goals of the training phase were:

* Use a distributed approach (TPU) to optimise training time
* Create a sample dataset for training
* Compare differents pre-trained model
* Optimise and tune the selected model
* Train the optimised model on the full dataset

Notebook => https://github.com/xavierbarbier/Plant_Pathology_2021_FGVC8/blob/main/plant-pathology-2021-fgvc8-training.ipynb



