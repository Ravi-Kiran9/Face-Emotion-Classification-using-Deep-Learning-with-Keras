# Face-Emotion-Classification-using-Deep-Learning-with-Keras
Classifying Face Emotions into three classes with Keras.

Keras is an open-source library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.

The data is a csv file of 10,000 48 x 48 images with pixel values in each column and its respective emotion. The emotions considered in this data are - Fear, Sad, Happy. 
This data is imported using pandas (a library for data manipulation and analysis) as a dataframe. The three labels are replaced with 0,1 and 2 respectively.
The data is splitted into X_train and y_train where X_train is the data of pixels and y_train is the labels.
The image dataframes are preprocessed using ImageDataGenerator.

# Model Architecture
Convulational Neural Networks are employed for working with images as they provide better results than other neural networks.
Conv2D layers along with BatchNormalisation and Dropout are used.
