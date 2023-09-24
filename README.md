**Introduction:**

The project focuses on predicting whether a client can repay a loan or not, using the “Home Credit Default Risk” Kaggle Competition dataset. The aim is to improve the loan experience for clients who have insufficient credit histories by using additional information, such as telco and transactional data. Following extensive exploratory data analysis(EDA), We performed feature engineering, introduced three new features, trained final features on various models such as Logistic Regression, Random Forest and Decision tree with hyper parameter tuning using gridsearch cv. For the final phase of the project, three neural networks were implemented using PyTorch Lightning for classification.The dataset was transformed into a tensor using the 'Data_pre_nn' function, and ReLU was used as the activation function, along with the Adam optimizer. The binary cross-entropy loss function was used, and backward propagation was implemented to minimize the loss. Neural Network 1, with 4 hidden layers, 400 epochs, and a batch size of 100, gave the best results. Neural Network 2 had the same architecture as Neural Network 1, but with 100 epochs and a batch size of 128. Neural Network 3 had a different architecture, with 3 hidden layers and 2 sigmoid output layers. The first neural network performed better compared to other 2 with accuracy of 0.938 and AUC score of 0.580

The major challenges we faced were choosing the correct architecture. In building an MLP architecture can effect the performance of the model hence choosing the correct parameters such as learning rate, number of epochs, optimizer and activation function plays a major role. To tackle this, we performed multiple experiments by changing the parameters and finally obtained the best AUC score. Along with this we also faced issues with computational complexity as training a neural network model can be time consuming especially with a greater number of epochs. 


**Workflow:**

![Group21_Phase4_Block_D.png](https://drive.google.com/uc?export=view&id=1malbxjOyJ6j_TRrkmmUQ8xcdysOq6L1R)


**Data Lineage:**

![Group21_Phase4_Block_D.png](https://drive.google.com/uc?export=view&id=1iPWN7OrwuNnFbRFsphVQzidNPtswQeh1)


**Neural Network:** 

we implemented three Neural Networks using PyTorch Lightning for classification. For the classification of neural network, we transformed the dataset into a tensor using the ‘Data_pre_nn’ function. This Neural Network has 160 input features and one output feature.   

The DataLoader function then converts the train, test and validation data with 160 features. We have used the ReLU (Rectified Linear Unit) function as the activation function and Adam algorithm as an optimizer with a learning rate of 0.01.   

For the loss function, we have used the Binary cross entropy loss function, and are using Backward Propagation to minimize the loss.  
                                                                $$BCE(t,p) = -(t\log(p) + (1-t)\log(1-p))$$ 
Neural Network 1 is implemented with 4 hidden layers, 400 epochs, 100 batch size and below is the string form of the architecture :   
160 - 128 - ReLU - 64 - ReLU - 32 - ReLU - 16 - ReLU - 1 sigmoid

Neural Network 2 is implemented with same 4 hidden layers but100 epochs, 128 batch size and below is the string form of the architecture :   
160 - 128 - Relu - 64 - Relu - 32 - Relu - 16 - Relu - 1 sigmoid

Neural Network 3 is implemented with 3 hidden layers, 100 epochs, 128 batch size and below is the string form of the architecture :   
160 - 128 - Relu - 64 - Relu - 32 - Relu - 2 sigmoid

Overall, the performance of the model depends on the architecture like learning rate, epochs, the choice of optimizer and loss function. In this case, Neural Network 1 with higher learning rate, more epochs, and with 4 hidden layers gave the best results. 



**Data Leakage:** 

Data Leakage occurs when the training dataset contains the information that we are going to predict, also if the data is taken from outside the training set. 

Steps we have taken to avoid data leakage:  

We split the data to create a Validation set and kept it held out before working on the model.  

Also, we performed Standardization were done on the train and test set separately, to avoid knowing the distribution of the entire dataset.  

We made sure that we did the correct evaluation by using the appropriate metrics, models and careful about the data leakage. Hence, we made sure we avoided any cardinal sins of ML.


**Pipeline:**
![model_pipeline.jpeg](https://drive.google.com/uc?export=view&id=1txAtnDo_JGSetl4IukCwqd8IdhZ478u9)


**Number of Experiments:**

In this phase we have conducted three Experiments for the neural network by changing the parameters. For all the Experiments we have used the adam optimizer and for Experiment 2 and 3 we took the learning rate of 0.001 and 100 epochs. For Experiment 3 we took 400 epochs and learning rate of 0.01. As per the hidden layers we have taken 4 for the 1st two experiments and 3 hidden layers for the last experiment. 


**Experiment Results:**
![picture](https://drive.google.com/uc?export=view&id=1O7KS9OMKeIHWgUFREwW4jTUMyi_tyetQ)

We have used the 160 features and implemented MLP model using PyTorch lightning in which we have conducted 3 experiments with different parameters. In the result table we have displayed various metrics such as accuracy, AUC, Log loss and also parameters such as optimizer, number of epochs, hidden layers. From the results table, Experiment 1 had the highest AUC_ROC score of 0.580125, followed by experiment 3 with an AUC_ROC score of 0.530666. Experiment 2 had the lowest AUC_ROC score of 0.5. In terms of other metrics, all three experiments had the same test accuracy of 0.918588, however, the train accuracy was highest in experiment 1 with a value of 0.9388, while experiments 2 and 3 had train accuracies of 0.9059 and 0.9412, respectively. Based on these results, we can infer that the first Experiment achieved the highest AUC-ROC score, indicating that it performed the best in predicting the binary output for the given dataset. However, the other two models also achieved a high test accuracy, indicating that they were also able to learn the underlying patterns in the data. We can see that Model 2 has achieved the highest train accuracy, which may suggest that it overfits to the training data. Finally, the different loss functions and hyperparameters used in the models led to the differences in their performance. 


**Conclusion:** 

The Home Credit Default Risk Initiative project aims to predict whether a borrower will repay their debt accurately. Our hypothesis follows: ML pipelines used with custom features or by taking only appropriate features instead of all features, might give good predictions for the client's repaying capability. In the phase1 we have performed EDA where we got lot of insights into data and their relations, In the next phase we focused on feature engineering where we used functions to drop columns with missing values and collinear features and created three new features based on domain knowledge. We then conducted hyperparameter tuning using grid search and cross-validation to find the best hyperparameters. After this we implemented an MLP model using PyTorch lightining where we have conducted 3 experiments by changing the parameters. In this we got the best results for Exp1 with highest AUC score of 0.58 and test accuracy of 0.918 when we used the following parameters: 400 epochs, learning rate as 0.01, adam optimizer, BCE loss for the loss function. Overall, we didn’t get expected result using the MLP model as the obtained AUC score is not ideal and less when compared to AUC score obtained from logistic regression model in previous phase. 
