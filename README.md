# Disease_Prophecy
## Description
In this project first the dataset is cleaned so that it will become workable. The dataset contain the name of disease with its symptoms and the weight associated with the disease.
In the cleaned dataset we will make the mapping of each disease with its symptoms. So, each disease is mapped with its symptoms and its corresponding weights. Now, the cleaned dataset
is used with Dataframes so that we can manipulate the dataset. The Dataframes in pandas library helps to manipulate the data in a 2D structure. So, the symptoms and diseases of the 
cleaned dataset are arranged in the tabular form. After this, we will separate the symptoms and diseases from this Dataframes and finally merge them to form the dataset on which we will
actually work. Now, each disease is present in the column and in the corresponding row each of the symptom is marked as 1 in the dataset. 
Now we try our classifier on this dataset. Here, we will use **MultinomialNB()** to classify our dataset, because **MultinomialNB()** is much better way of classification if the input dataset
is present in the textual format. Here we have to note that classifier don't work on the unseen data so if we partition the data in the training and testing set then we don't get our
appropriate result as we can't predict diseases which it hasn't seen before. So, we have to train the classifier on the total dataset. This is again Multilabel classification, here we
got the score of 90% for the prediction values. Now we will train our ***DecisionTreeClassifier*** which is main classifier which we use to classify our disease list and for the prediction of
diseases.

### **Why DecisionTreeClassifier is used here ?**

-> Able to handle multi-output problems.

-> Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.

->Simple to understand and to interpret. Trees can be visualised.

We have also done analysis on Manual Training dataset collected similar to the training of previous dataset. Now we have also added **cross-validation** after the classifier . And the cross-validation
score we got are **97.52%** for our DecisionTreeClassifier.

### **Why we are doing cross-validation here ?**

Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data.

To avoid **over-fitting**, we have to define two different sets : a training set **X_train, y_train** which is used for learning the parameters of a predictive model, and a testing set **X_test, y_test** which is used for evaluating the fitted predictive model.

However, by defining these two sets, we drastically reduce the number of samples which can be used for learning the model, and the results can depend on a particular random choice for the pair of (train, test) sets.

**A solution is to split the whole data several consecutive times in different train set and test set**, and to return the averaged value of the prediction scores obtained with the different sets. Such a procedure is called **cross-validation**. This approach can be computationally expensive, but does not waste too much data (as it is the case when fixing an arbitrary test set), which is a major advantage in problem such as inverse inference where the number of samples is very small.

Reference : http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/cross_validation.html

After all this now, we will find the important features actually symptoms which helps in the classification through DecisionTreeClassifier method. For this, we will store the
importance of each features and then sort in descending order . Then we found the feature having highest importance. Actually, this feature form the root of the Decision Trees
which we used as our classifier. this feature(symptom) has the high potential to classify all the symptoms for their corresponding diseases. So, Now we can use our classifier to 
predict any disease.

# Note 
The full version of Disease_Prediction_From_Symptoms.ipynb file can be seen at below link. 
http://nbviewer.jupyter.org/github/victor-ludorum/Disease_Prophecy/blob/master/Disease_Prediction_From_Symptoms.ipynb
