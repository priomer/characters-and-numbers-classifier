This project is done for the task submission under MIDAS@IIITD Summer Internship.
# Task taken-> Task-2
#### This Task is divided in three parts:
1. Using the [dataset](https://www.dropbox.com/s/pan6mutc5xj5kj0/trainPart1.zip) train a CNN and use no other data source or pretrained networks.
2. This part has further two parts applicable to a subset of previous dataset.
 - By selecting only 0-9 training images from the above dataset use the pretrained network to train on MNIST dataset.
 - Using the standard MNIST train and test splits compare with the above results.
3. Finally, take the following [dataset](https://www.dropbox.com/s/otc12z2w7f7xm8z/mnistTask3.zip), and train on this dataset.

## Task-2 part-1
- The dataset has been taken from [here](https://www.dropbox.com/s/pan6mutc5xj5kj0/trainPart1.zip).
- The dataset has 62 labels with 10 labels of 0 to 9 digits and 26-26 labels for small and capital letters of english alphabets.
- Each label has 40  images in total,which I have splitted as training and validaion images in the code itself.
- The code for training the CNN model is included in this repository.
- I have followed a very simple symmetrical sequential model with hidden layers after certain convolution layers.
- I have taken reference from my earlier CNN model which was plant leaf disease classifier.
- The code is the most optimized model which gives 90.01% accuracy on training.
- All the specifications of each code block used is mentioned as comments in the file itself.
- The corresponding file for this task has been uploaded by the name midas_task1.py

## Task-2 part-2
This part has further two subparts:
- Subset of previous dataset 0 to 9 digit only: 
   - At first I have used the same pretrained neural net same as for part-1 and used a subset of the previous dataset.
   - This time since the number of categories are lesser so we can decrease the number of epochs and the number of layers too, so the final optimized result has been achieved with an accuracy of 92.6% and the corresponding code has been uploaded by the name midas_task2_0to9.py
- Using the standard [mnist_dataset](http://yann.lecun.com/exdb/mnist/):
   - Trained the same model and since this dataset has 60000 images distributed among 10 labels so more effective results have been obtained and an accuracy of 97% has been achieved, as compared to only 40 images per labels in the previous dataset.

## Task-2 part-3
- Using the same model as above, this time a different dataset, which be downloaded from [here](https://www.dropbox.com/s/otc12z2w7f7xm8z/mnistTask3.zip) is used.
- This dataset has shuffled 60000 images of digits from 0 to 9 distributed randomly among 10 labels.
- Due to randomly distributed images and no proper classification in the dataset the model could not be trained effectively and this can also be analyzed qualitatvely.  
