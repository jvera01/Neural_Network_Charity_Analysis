# Neural_Network_Charity_Analysis
![](resources/Banner1.PNG)
# Overview & Purpose

Bek’s come a long way since her first day at that boot camp five years ago—and since earlier this week, when she started learning about neural networks! Now, she is finally ready to put her skills to work to help the foundation predict where to make investments.

With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special consideration for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively



# Results

## Deliverable 1: Preprocessing Data for a Neural Network Model
Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
What variable(s) are considered the target(s) for your model?
What variable(s) are considered the feature(s) for your model?
Drop the EIN and NAME columns.
Determine the number of unique values for each column.
For those columns that have more than 10 unique values, determine the number of data points for each unique value.
Create a density plot to determine the distribution of the column values.
Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, Other, and then check if the binning was successful.
Generate a list of categorical variables.
Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame.
Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals.

At this point, your merged DataFrame should look like this:

![](resources/D1.PNG)

## Deliverable 2: Compile, Train, and Evaluate the Model
Follow the instructions below and use the information file to complete Deliverable 2.

Continue using the AlphabetSoupCharity.ipynb file where you’ve already performed the preprocessing steps from Deliverable 1.
Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
Create the first hidden layer and choose an appropriate activation function.
If necessary, add a second hidden layer with an appropriate activation function.
Create an output layer with an appropriate activation function.
Check the structure of the model.
Compile and train the model.
Create a callback that saves the model's weights every 5 epochs.
Evaluate the model using the test data to determine the loss and accuracy.
Save and export your results to an HDF5 file, and name it AlphabetSoupCharity.h5.

Save your AlphabetSoupCharity.ipynb file and AlphabetSoupCharity.h5 file to your Neural_Network_Charity_Analysis folder.

![](resources/D2.PNG)

## Deliverable 3: Optimize the Model
Follow the instructions below and use the information file to complete Deliverable 3.

Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.
Import your dependencies, and read in the charity_data.csv to a Pandas DataFrame.
Preprocess the dataset like you did in Deliverable 1, taking into account any modifications to optimize the model.
Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
Create a callback that saves the model's weights every 5 epochs.
Save and export your results to an HDF5 file, and name it AlphabetSoupCharity_Optimization.h5.
Save your AlphabetSoupCharity_Optimzation.ipynb file and AlphabetSoupCharity_Optimization.h5 file to your Neural_Network_Charity_Analysis folder.
## Compiling, Training, and Evaluating the Model

### Test Version 1
The first time trying the neuron network I used 8 neurons in the first layer. I used 6 in the second. In both layers had relu activations fuctions. Below you can find the the peformance mentrics of the first model. 

![](resources/TraningApp.PNG)

### Test Version 2
In this model, the classification column seems to be the most important. It is also interesting to note that the input layer is not doing much. It seems like the model is not able to reweight the input from the second layer to the third layer.

I wanted to see if I can improve the performance of this model by adding a fourth layer. I also changed the activation function of the first layer to ReLU and changed the activation function of the third layer to Softmax.

![](resources/TrainingV1.PNG)

### Test Version 3
This model performed better than the one above. I am not sure if it is because of the tanh activation function or the thresholding. I will try different combinations of activation functions and thresholding values to see if I can find a combination that performs better.  In my fourth iteration, I changed the activation function for the three layers to sigmoid. Here are the performance metrics for this model:

![](resources/TrainingV2.PNG)

### Test Version 4
I got a slightly better result this time. I wonder if I can improve the accuracy of this model by changing the parameters of the network.  I am going to change the number of neurons in the first layer from 12 to 10 and train the model again.  Here are the performance metrics of this model.  This is a slight improvement from the previous model. I will now try to change the activation function from relu to tanh.  Here are the performance.

![](resources/TrainingV3.PNG)


# Summary

I was not able to achieve the accuracy of my neural network of 75%. I want to change the activation function, change the number of layers, and change the number of neurons in each layer. I would also like to see if I can change the dataset to better fit, my model. 

I was able tro upload the following to your Neural_Network_Charity_Analysis GitHub repository:

- Your AlphabetSoupCharity.ipynb file for Deliverables 1 and 2.
- Your AlphabetSoupCharity.h5 file for Deliverables 1 and 2.
- Your AlphabetSoupCharity_Optimzation.ipynb file for Deliverable 3.
- Your AlphabetSoupCharity_Optimzation.h5 file for Deliverable 3.
- An updated README.md that has your written report.

I was able to learn the following topics. 
- Compare the differences between the traditional machine learning classification and regression models and the neural network models.
- Describe the perceptron model and its components.
- Implement neural network models using TensorFlow.
- Explain how different neural network structures change algorithm performance.
- Preprocess and construct datasets for neural network models.
- Compare the differences between neural network models and deep neural networks.
- Implement deep neural network models using TensorFlow.
- Save trained TensorFlow models for later use.
