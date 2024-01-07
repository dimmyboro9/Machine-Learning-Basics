# Machine Learning Basics
There is a repository showing my first steps in machine learning and containing my first projects, which I created during my studies at FIT CTU. It demonstrates my knowledge in working with Python libraries such as Pandas, NumPy, Matplotlib, Scikit-learn, PyTorch and my general understanding of the principles of machine learning and some of its models.

## Projects
- [Dimensionality reduction and binary classification](#Dimensionality-reduction-and-binary-classification)
- [Neural networks](#Neural-networks)
- [Clustering](#Clustering)
- [Regression](#Regression)

## Dimensionality reduction and binary classification
This machine learning project involves binary classification and dimensionality reduction using a dataset loaded from [Fashion Mnist dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist). The primary objectives are to build, evaluate, and optimize classification models while exploring various dimensionality reduction techniques.

The project begins with data loading and splitting into three subsets, followed by essential data exploration to gain insights and visualize patterns. After that applied three classification models: **Support Vector Machine (SVM)**, **Naive Bayes classifier**, and **Linear Discriminant Analysis (LDA)**, each with its unique attributes and hyperparameters, on the dataset I had after data preprocessing. Model selection, hyperparameter tuning, and data standardization are integral parts of the process.

Then, dimensionality reduction methods, **Principal Component Analysis (PCA)** and **Locally Linear Embedding (LLE)**, are employed to improve model efficiency. The project concludes by selecting the best-performing model and estimating its accuracy on new, unseen data.

## Neural networks
This machine learning project is dedicated to image multiclass classification, where the goal is to classify images into multiple categories using neural networks. The project begins with loading image data from [Fashion Mnist dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist), data handling and organizing it into subsets for training, model comparison, and performance prediction. Then, data exploration techniques are employed to uncover patterns, detect outliers, and understand the dataset's characteristics. Data visualization, including graphs and images, is used for better understand of our exploration conclusions.

The heart of the project revolves around the development and training of various neural network models. The project includes the creation of different **Feedforward Neural Network (FNN)**, **Convolutional Neural Networks (CNN)** and their **combinations**. Each of the models subjected to extensive experimentation. This experimentation includes architectural variations such as layer depths and sizes, data standardization/normalization, optimization methods, and regularization techniques. When creating combinations of feed-forward and convolutional neural networks, I also experimented with the overall network architecture and the order of layers. The suitability of each model for multiple classification is evaluated using model evaluation methods, and the results are accompanied by detailed comments. The project concludes by selecting the best-performing model and estimating its accuracy on new, unseen data.

## Clustering
This project delves into the analysis of credit card data using **clustering techniques**. The dataset, sourced from the file `CC GENERAL.csv` (available [here](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)), provides information about credit card usage. To comprehend the dataset's intricacies, the initial steps involve thorough data exploration, uncovering patterns, and addressing issues like missing values. The dataset's description in `data_description.txt` guides this exploration, offering context for the variables used.

A pivotal aspect of this project is the creation of a **KMeans algorithm**, followed by the application of clustering using both the custom implementation and the `sklearn.cluster.KMeans` and comparing results between them. Then I perform agglomerative hierarchical clustering and plot dendrograms for visual comprehension. Different experimentation includes tuning hyperparameter values and using data standardization. The **Silhouette method** helps me in evaluating cluster quality.

The ultimate goal is to choose the most fitting clustering approach. The selected method helps understand and group credit card users by looking at things like `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `PAYMENTS`, and more. This helps identify different types of users in the dataset.

This project was implemented on 28.12.2022.

## Regression
This project focuses on analyzing life expectancy data sourced from the file `LifeExpectancyData.csv`, available [here](https://www.kaggle.com/kumarajarshi/life-expectancy-who). The dataset's description, accessible on the original dataset page, guides the exploration process. 

The initial step involves dividing the data into appropriate subsets. Then, we carefully explore the data, paying attention to things like missing information. Based on what we see, we take actions to make sure the data is reliable and complete.

**Linear** and **ridge regression** models are applied, with a thorough evaluation using **MAE** (mean absolute error) and **RMSE** (root mean squared error) for error measurement. To increase models performance, experiments include data standardization and tuning hyperparameters. In addition to linear and ridge regression, other models are explored, broadening the analysis. The final step involves selecting the best-performing model based on RMSE and estimating the expected RMSE and MAE on new, previously inaccessible data, providing insights into the model's predictive capabilities.

This project was implemented on 4.12.2022.

`Copyright (c) Dmytro Borovko 2023`