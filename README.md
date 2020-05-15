# Lifelong-learning-neural-models-for-streaming-classification
This repository stores all my code from the Independent Study that I am pursuing while doing my Masters' at the Rochester Institute of Technology as a member of the Neural Adaptive Computing Laboratory, advised under Dr. Alex Ororbia. <br/>
**NOTE**: This work is currently ongoing and is not the final version. Therefore, this README keeps a track of progress so far.


## Data Loading
For beginning, I am using Caltech-256 as the dataset for this work. Following is the way to structure the raw dataset so that it could be used in a Convolutional Neural Network that is developed further.
1. Download the dataset from http://www.vision.caltech.edu/Image_Datasets/Caltech256/
2. Run the split.py script in the decompressed folder
3. Delete all the empty folders now except the 'train' and 'test' folders which now contains a split of the entire dataset

### Code details
- mlp-v1.2.ipynb - working code for observing forgetting in mnist and fashionmnist
