# Lifelong-learning-neural-models-for-streaming-classification
This repository stores all my codes from the Independent Study that I am pursuing while doing my Masters' at the Rochester Institute of Technology <br/>
**NOTE**: This work is currently ongoing and is not the final version. Therefore, this README keeps a track of progress so far.


## Data Loading
For beginning, I am using Caltech-256 as the dataset for this work. Following is the way to structure the raw dataset so that it could be used in a Convolutional Neural Network that is developed further.
1. Download the dataset from http://www.vision.caltech.edu/Image_Datasets/Caltech256/
2. Run the split.py script in the decompressed folder
3. Delete all the empty folders now except the 'train' and 'test' folders which now contains a split of the entire dataset

### Week1.ipynb
This notebook contains code for a Convolutional Neural Network written in PyTorch.
