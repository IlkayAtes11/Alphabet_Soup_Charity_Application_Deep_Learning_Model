### Alphabet Soup Charity Application Deep Learning Model

#### Overview of the analysis: 

The analysis used real world dataset which contains more than 35k organizations` data, funded by Alphabet Soup in the past. With the help of this data, the aim of the analysis is to create a deep learning neural network model to predict the success/failure of the future funds.

#### Results: 

##### Data Preprocessing

* The column named "IS_SUCCESSFUL" is target of this model (dependent variable)  
* During optimization attempts, different feature sets were used to understand their effects on the loss/accuracy.
* For all optimization attempts "EIN" and "NAME" variables were removed.

##### Compiling, Training and Evaluating the Model

* How many neurons, layers, and activation functions did you select for your neural network model, and why?

Main neural network model has contained two hidden layers with 80 and 30 neurons respectively. The hidden layes used "relu"activation function while the output layer used 'sigmoid'activation function. In this analysis, the epochs number was 100. 

* Were you able to achieve the target model performance?

The original neural network model could not achieve the target model performance which is 75%. The accuracy of this model was approximately 73% and the loss was 56%.

* What steps did you take in your attempts to increase model performance?

  Three different attempts were made to optimize the model:

    * Attempt #1
      In this attempt, couple of different alterations were made to model. Two more variables were dropped ('STATUS' and 'SPECIAL_CONSIDERATION'), binning structures         of 'APPLICATION_TYPE' and 'CLASSIFICATION' variables have been changed, a new hidden layer was added with 60 neurons. After all these changes, accuracy level           stayed  same (73%) while loss level increase (58%).

    * Attempt #2
      In this attempt, number of epoches was increased to 150 from 100.The new hidden layer that was added in attempt 1 stayed same. After change in the epch numbers, the accuracy level stayed same (73%) while loss level increase (58%).The previous attempt didn't get us over the 75% threshold for accuracy we would like to acheive. Maybe more neurons, and a third hidden layer will do the trick.     I've also swapped the activation functions to tanh and sigmoid to see if these will help with our classification accuracy

    * Attempt #3
     Perhaps more layers and neurons will get us a higher accuracy. Added in 2 more layers for a total of 5 hidden layers.Optimizing the neural network model to make the attempt to reach target model performance, I have increased the amount of neurons in the first hidden layer from 80 to 100. I have also increased the amount of neurons the second hidden layer from 30 to 60. Making these changes resulted in a target performance of 71% which is closer to the target model. However, I did make attempts to adding another hidden layer with varying neuron values which resulted in lower accuracy ratings.


#### Summary: 

The optimization model as a result reached a peak of 71% which is relatively close to the target performance model and surpassed the initial model. The loss in accuracy from making other attempts such as adding in more hidden layers could be the result of the model being over fit. The model could have also become more accurate by potentially removing more features from the dataset and also adding more data. A different model that could have been used is possibly the Random Forest Classifier which could have avoided the data from being over fit. The Random Forest Classifier is also robust and could be deemed to be a more accurate model because of its sufficient number of estimators and tree depth.

In the Alphabet Soup Challenge workbook 3 different binary classification neural networds were applied to the data.

For the three neural networks I started with a "shallow" deep neural network with 2 hidden layers with 24 neurons. The initial number of neurons and hidden layers is chosen arbitrarily. The model acheived an accuracy of 72.23%, and establishes the baseline accuracy to improve on. The next two attempts to build a better model I increased the number of hidden layers and neurons per layer drastically. The second attempt has 3 hidden layers, with the layers having 50 or 100 neurons per layer. The second layer boasts an accuracy of 72.39%, and does not improve on the previous model. The third attempt I went with with 5 hidden layers with the same 50/100 neurons per layer. This model I thought would overfit the data and give an accuracy better than the second attempt, and it did. The third attempt achieved an accuracy of 72.43% and does not improve the previous models.

I was not able to achieve the target model performance of 75%. Through each attempt I attempted adding/removing hidden layers and neurons. I also changed the activation functions for many of the hidden layers, and settled on the sigmoind activation function for the outer layer gave the best performance. There are issues with using the rectified linear unit (reLU) and leaky reLU methods for this dataset. If I were to implement a different model it would be worthwhile to explore ensemble trees/random forests with this dataset. There are a lot of categorical variables that a tree model may do well on. Perhaps the tree model can capture more of the "quirks" of the data versus the neural networks getting hung up on trying to find paths of relations in the dataset.










