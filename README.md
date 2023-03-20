# Deep_Learning_Challenge

Attempt #1
This is a deep neural network with 2 hidden layers, and rectified linear unit activation functions. The output layer has a sigmoid activation function

Attempt #2
The previous attempt didn't get us over the 75% threshold for accuracy we would like to acheive. Maybe more neurons, and a third hidden layer will do the trick. I've also swapped the activation functions to tanh and sigmoid to see if these will help with our classification accuracy

Attempt #3
Perhaps more layers and neurons will get us a higher accuracy. Added in 2 more layers for a total of 5 hidden layers.

In the Alphabet Soup Challenge workbook 3 different binary classification neural networds were applied to the data.

For the three neural networks I started with a "shallow" deep neural network with 2 hidden layers with 24 neurons. The initial number of neurons and hidden layers is chosen arbitrarily. The model acheived an accuracy of 72.23%, and establishes the baseline accuracy to improve on. The next two attempts to build a better model I increased the number of hidden layers and neurons per layer drastically. The second attempt has 3 hidden layers, with the layers having 50 or 100 neurons per layer. The second layer boasts an accuracy of 72.39%, and does not improve on the previous model. The third attempt I went with with 5 hidden layers with the same 50/100 neurons per layer. This model I thought would overfit the data and give an accuracy better than the second attempt, and it did. The third attempt achieved an accuracy of 72.43% and does not improve the previous models.

I was not able to achieve the target model performance of 75%. Through each attempt I attempted adding/removing hidden layers and neurons. I also changed the activation functions for many of the hidden layers, and settled on the sigmoind activation function for the outer layer gave the best performance. There are issues with using the rectified linear unit (reLU) and leaky reLU methods for this dataset. If I were to implement a different model it would be worthwhile to explore ensemble trees/random forests with this dataset. There are a lot of categorical variables that a tree model may do well on. Perhaps the tree model can capture more of the "quirks" of the data versus the neural networks getting hung up on trying to find paths of relations in the dataset.



Alphabet Soup Challenge
Build your own machine learning model that will be able to predict the success of a venture paid by Alphabet soup. Train model will be used to determine the future decisions of the company—only those projects likely to be a success will receive any future funding from Alphabet Soup.

Project Overview
Objectives
The goals of this challenge:

Import, analyze, clean, and preprocess a “real-world” classification dataset.
Select, design, and train a binary classification model of your choosing.
Optimize model training and input data to achieve desired model performance.
Resources
Data Source: charity_data.csv Software: Python 3.7.6, Anaconda 4.8.4, Jupyter Notebook 6.0.3

Challenge Overview
Create a new Jupyter Notebook within a new folder on your computer. Name this new notebook file “AlphabetSoupChallenge.ipynb” entifiable).

Download charity_data.csv

Import and characterize the input data.

What variable(s) are considered the target for your model? SPECIAL_CONSIDERATIONS and IS_SUCCESSFUL

What variable(s) are considered to be the features for your model? APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, IS_SUCCESSFUL

What variable(s) are neither and should be removed from the input data? EIN and NAME

Using the methods described in this module, preprocess all numerical and categorical variables, as needed:

Combine rare categorical values via bucketing.
Encode categorical variables using one-hot encoding.
Standardize numerical variables using Scikit-Learn’s StandardScaler class
Using a TensorFlow neural network design of your choice, create a binary classification model that can predict if an Alphabet Soup funded organization will be successful based on the features in the dataset.

You may choose to use a neural network or deep learning model.
Compile, train, and evaluate your binary classification model. Be sure that your notebook produces the following outputs:

Final model loss metric
Final model predictive accuracy
Do your best to optimize your model training and input data to achieve a target predictive accuracy higher than 75%.

Create a new README.txt file within the same folder as your AlphabetSoupChallenge.ipynb notebook. Include a 5–10 sentence writeup in your README that addresses the following questions:

How many neurons and layers did you select for your neural network model? Why?
I used Neural network model to predict the success of an organization. Tried not to overfit the model, so I didn't use a lot of layers. 1 input and 2 hidden_nodes_layer(units) were 16 and 1. Another input and 3 hidden_nodes_layer(units) were 16, 8, and 1. The number of epochs for each were 100.

Were you able to achieve the target model performance? What steps did you take to try and increase model performance?
I was not successful at achieving 75% when my target was IS_SUCCESSFUL. I got an average of 72.5%. I changed several unit/neurons and the accuracy didn't improve. Therefore, I decided to change the target. Changing the target jumped the accurary to 99%.

If you were to implement a different model to solve this classification problem, which would you choose? Why?
SVM model are less prone to overfitting because they are trying to maximize the distance, instead of enclose all data within a boundary.

Alphabet Soup Deep Learning
Analysis Overview
The analysis used machine learning and neural networks to predict whether if the applicants funded by the charity organization known as Alphabet Soup will be successful or not. The prediction resulted from the creation of a binary classifier that analyzed a dataset containing 30,000+ organizations funded by Alphabet Soup. In this project, it comprised of:

Preprocessing data for the neural network
Compiling, training, and evaluating the model
Optimizing the model
Examples
-Preprocessing Model



-Optimization Model



Results
Data Preprocessing
Variable that I considered as the target in my model is the 'IS_SUCCESSFUL' column.
Variables that were considered as the features in my model is every other column except for the 'IS_SUCCESSFUL' column and everything else that has been dropped.
Variables that were considered to be neither targets or features are the 'EIN' and 'NAME' columns as they would have no significant value to the analysis.
Compiling, Training, and Evaluating the Model
For my neural network model, it contained two hidden layers where my first layer had 80 neurons and the second layer had 30 neurons. There is also an output layer. Both my first and second hidden layers have the 'relu' activation and the activation of the output layer is 'sigmoid'.
Unfortunately the neural network model was unable to reach the target model performance of 75%. The accuracy of my model turned out to be around 69%.
Optimizing the neural network model to make the attempt to reach target model performance, I have increased the amount of neurons in the first hidden layer from 80 to 100. I have also increased the amount of neurons the second hidden layer from 30 to 60. Making these changes resulted in a target performance of 71% which is closer to the target model. However, I did make attempts to adding another hidden layer with varying neuron values which resulted in lower accuracy ratings.
Summary
The optimization model as a result reached a peak of 71% which is relatively close to the target performance model and surpassed the initial model. The loss in accuracy from making other attempts such as adding in more hidden layers could be the result of the model being over fit. The model could have also become more accurate by potentially removing more features from the dataset and also adding more data. A different model that could have been used is possibly the Random Forest Classifier which could have avoided the data from being over fit. The Random Forest Classifier is also robust and could be deemed to be a more accurate model because of its sufficient number of estimators and tree depth.
