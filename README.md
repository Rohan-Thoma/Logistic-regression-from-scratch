# How to implement logistic regression from scratch with L2 regularization without using sklearn

![Screenshot 2023-10-04 142000_x16_drawing](https://github.com/Rohan-Thoma/Logistic-regression-from-scratch/assets/98249384/cf9753cf-d3a7-4307-b3f5-8e4d4baebbdc)

<br>

## 🎯 Objective 
To train and test logistic regression model without using any external libraries

➡️ Logistic regression is one of the first machine learning algorithms that anyone learns about when getting started with AI and ML. Understanding this alogorithm inside out is most important to guage the depth of knowledge of a machine learning engineer or a data scientist. <br>
➡️ Knowledge of how to implement from scratch will be very useful while deploying the machine learning models in the production environment where we cannnot depend on external libraries which are not at all optimized to tackle real world data in raw state. We need to know the algorithm inside out to write in other languages such as C++ etc which are faster than python. <br>
➡️ In this notebook we will see how to implement logistic regression step by step and also compare our results and weights with scikit learn for checking the credibility of the solution.

## ☕ Procedure
### 🔶 1. Creating a custom dataset
As this is an implementation walkthorugh, we will just create a dummy classifcation dataset for this notebook. After creating the dataset we will also split the dataset in to train and test sets.

### 🔶 2. Training the sklearn model for comparision of results
Here we train the sklearn LR model to get the weights and biases and save them for later for comparision with our custom implementation.

### 🔶 3. Implement Logistic regresion with L2 regularization without using sklearn
#### 🏋️‍♂️ 3.1 Initialize the weights
Here we write a function to initialize the weights and bias terms

#### 💻 3.2 Compute the sigmoid function
$sigmoid(z)= 1/(1+exp(-z))$ <br>
Here we write a function to implement the sigmoid function with the above given formula

#### 📉 3.3 Compute log-loss
$log loss = -1*\frac{1}{n}\Sigma_{for each Y_{t},Y_{pred}}(Y_{t}*log_{10}(Y_{pred})+(1-Y_{t})*log_{10}(1-Y_{pred}))$ <br>
Here we write a function to compute the log-loss value with the above given formula.

#### 🛼 3.4 Compute the gradient wrt 'w' (weights) 
$dw^{(t)} = x_n(y_n − σ((w^{(t)})^{T} x_n+b^{t}))- \frac{λ}{N}w^{(t)}$ <br>
Here we write a function to compute the value of gradiets wrt weights with the help of above formula.

#### ⚽ 3.5 Compute the gradient wrt 'b' (bias)
$db^{(t)} = y_n- σ((w^{(t)})^{T} x_n+b^{t})$ <br>
Here we write a function to compute the value of the gradient wrt to the bias term with the help of the above given formula.

#### 🖖3.6 Prediction function
Here we write a function which takes in the model object and the dataset and outputs the results in the form of class labels for all the given data points.

### 🔶 4. Implementing the LR and training the model with the help of gradient descend algorithm
Here we write a function train() which is very similar to fit() in sklearn. This takes in the train and test datasets along with weights and regularization parameter intializations. We also take in the number of epochs and update the weights and biases after each epoch of training with the help of gradient descend algorithm. Here we also store the train and test losses for each epoch to plot later for visualization.

### 🔶 5. Comparision between the weights of sklearn and our custom implementation
Here we simply print the values of the weights and biases of the both the implementations and then also print the differences between the values to show that it is very negligible and close to zero.

### 🔶 6. Plot of train and test losses vs epochs to check for convergence 
Here we taek the arrays that we saved while training which contain the -log-loss values of train and test datasets and plot them to visualize the convergence.

## 




