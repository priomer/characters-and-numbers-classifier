This project is done for the task submission under MIDAS@IIITD Summer Internship.
####This Task is divided in three parts:
1. Using the [dataset](https://www.dropbox.com/s/pan6mutc5xj5kj0/trainPart1.zip) train a CNN and use no other data source or pretrained networks.
2. This part has further two parts applicable to a subset of previous dataset.
 - By selecting only 0-9 training images from the above dataset use the pretrained network to train on MNIST dataset.
 - Using the standard MNIST train and test splits compare with the above results.
3. Finally, take the following [dataset](https://www.dropbox.com/s/otc12z2w7f7xm8z/mnistTask3.zip), and train on this dataset.

## Task-1
- The code for training the CNN model is included in this repository.
- I have followed a very simple symmetrical sequential model with hidden layers after certain convolution layers.
- I have taken reference from my earlier CNN model which was plant leaf disease classifier.
- The code is the most optimized model which gives 90.01% accuracy on training.
