# Machine Learning Basics
There is a repository showing my first steps in machine learning and containing my first projects, which I created during my studies at FIT CTU. It demonstrates my knowledge in working with Python libraries such as Pandas, NumPy, Matplotlib, Scikit-learn, PyTorch and my general understanding of the principles of machine learning and some of its models.

## Projects
- [Dimensionality reduction and binary classification](#Dimensionality-reduction-and-binary-classification)
- [Neural networks](#Neural-networks)

## Dimensionality reduction and binary classification
This machine learning project involves binary classification and dimensionality reduction using a dataset loaded from [Fashion Mnist dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist). The primary objectives are to build, evaluate, and optimize classification models while exploring various dimensionality reduction techniques.

The project begins with data loading and splitting into three subsets, followed by essential data exploration to gain insights and visualize patterns. After that applied three classification models: **Support Vector Machine (SVM)**, **Naive Bayes classifier**, and **Linear Discriminant Analysis (LDA)**, each with its unique attributes and hyperparameters, on the dataset I had after data preprocessing. Model selection, hyperparameter tuning, and data standardization are integral parts of the process.

Then, dimensionality reduction methods, **Principal Component Analysis (PCA)** and **Locally Linear Embedding (LLE)**, are employed to improve model efficiency. The project concludes by selecting the best-performing model and estimating its accuracy on new, unseen data.

## Neural networks
This machine learning project is dedicated to image multiclass classification, where the goal is to classify images into multiple categories using neural networks. The project begins with loading image data from [Fashion Mnist dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist), data handling and organizing it into subsets for training, model comparison, and performance prediction. Then, data exploration techniques are employed to uncover patterns, detect outliers, and understand the dataset's characteristics. Data visualization, including graphs and images, is used for better understand of our exploration conclusions.

The heart of the project revolves around the development and training of various neural network models. The project includes the creation of different **Feedforward Neural Network (FNN)**, **Convolutional Neural Networks (CNN)** and their **combinations**. Each of the models subjected to extensive experimentation. This experimentation includes architectural variations such as layer depths and sizes, data standardization/normalization, optimization methods, and regularization techniques. When creating combinations of feed-forward and convolutional neural networks, I also experimented with the overall network architecture and the order of layers. The suitability of each model for multiple classification is evaluated using model evaluation methods, and the results are accompanied by detailed comments. The project concludes by selecting the best-performing model and estimating its accuracy on new, unseen data.

`Copyright (c) Dmytro Borovko 2023`