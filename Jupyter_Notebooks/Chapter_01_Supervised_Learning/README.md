You should work your way through the notebooks in the following order:


# [1. Linear Regression](https://github.com/SharifiZarchi/Introduction_to_Machine_Learning/blob/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/01-Linear%20Regression/01-Linear_Regression.ipynb)
In this notebook we solve the problem of Regression from scratch using only numpy. We also experiment with Polynomial Regression and use sklearn to solve real-world problems.

**Warning!** If you want to run this notebook online, make sure you run the following code to download the dataset:
```
!kaggle datasets download -d mokar2001/house-price-tehran-iran
!unzip house-price-tehran-iran.zip -d ./assets
```
And change the excel loading:
```
file_path = './assets/housePrice.xlsx'
df = pd.read_excel(file_path)
```
To csv loading:
```
file_path = './assets/housePrice.csv'
df = pd.read_csv(file_path)
```
Or alternatively, you could just run the following code:
```
!mkdir -p ./assets
!wget -O ./assets/housePrice.xlsx https://raw.githubusercontent.com/SharifiZarchi/Introduction_to_Machine_Learning/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/01-Linear%20Regression/assets/housePrice.xlsx
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharifiZarchi/Introduction_to_Machine_Learning/blob/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/01-Linear%20Regression/01-Linear_Regression.ipynb)
[![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/SharifiZarchi/Introduction_to_Machine_Learning/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/01-Linear%20Regression/01-Linear_Regression.ipynb)



# [2. Linear Classification](https://github.com/SharifiZarchi/Introduction_to_Machine_Learning/blob/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/02-Linear%20Classification/02-Linear_Classification.ipynb)
In this notebook we use Regression to solve binary classification problems and understand their limitations.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharifiZarchi/Introduction_to_Machine_Learning/blob/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/02-Linear%20Classification/02-Linear_Classification.ipynb)
[![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/SharifiZarchi/Introduction_to_Machine_Learning/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/02-Linear%20Classification/02-Linear_Classification.ipynb)



# [3. Logistic Regression](https://github.com/SharifiZarchi/Introduction_to_Machine_Learning/blob/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/03-Logistic%20Regression/03-Logistic_Regression.ipynb)
In this notebook we learn about Logistic Regression and repurpose Regression to classify data from small datasets.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharifiZarchi/Introduction_to_Machine_Learning/blob/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/03-Logistic%20Regression/03-Logistic_Regression.ipynb)
[![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/SharifiZarchi/Introduction_to_Machine_Learning/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/03-Logistic%20Regression/03-Logistic_Regression.ipynb)



# [4. K-Nearest Neighbors](https://github.com/SharifiZarchi/Introduction_to_Machine_Learning/blob/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/04-kNN/04-kNN.ipynb)
In this notebook we implement the kNN algorithm from scratch and use it for classification. 


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharifiZarchi/Introduction_to_Machine_Learning/blob/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/04-kNN/04-kNN.ipynb)
[![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/SharifiZarchi/Introduction_to_Machine_Learning/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/04-kNN/04-kNN.ipynb)



# [5. Ensemble Learning](https://github.com/SharifiZarchi/Introduction_to_Machine_Learning/blob/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/05-Ensemble%20Learning/05-Ensemble_Learning.ipynb)
In this notebook we implement the Decision Tree Classifiers from scratch and explore how ensembling different models improve classification. 
We also work with Random Forests and XGBoost and compare their results.


**Warning!** If you want to run this notebook online, make sure you run the following code to download the dataset:
```
!mkdir -p ./assets/imbalanced_datasets
!wget -O ./assets/imbalanced_datasets/1.csv https://raw.githubusercontent.com/SharifiZarchi/Introduction_to_Machine_Learning/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/05-Ensemble%20Learning/assets/imbalanced_datasets/1.csv
!wget -O ./assets/imbalanced_datasets/2.csv https://raw.githubusercontent.com/SharifiZarchi/Introduction_to_Machine_Learning/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/05-Ensemble%20Learning/assets/imbalanced_datasets/2.csv
!wget -O ./assets/imbalanced_datasets/3.csv https://raw.githubusercontent.com/SharifiZarchi/Introduction_to_Machine_Learning/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/05-Ensemble%20Learning/assets/imbalanced_datasets/3.csv
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharifiZarchi/Introduction_to_Machine_Learning/blob/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/05-Ensemble%20Learning/05-Ensemble_Learning.ipynb)
[![Open In kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/SharifiZarchi/Introduction_to_Machine_Learning/main/Jupyter_Notebooks/Chapter_01_Supervised_Learning/05-Ensemble%20Learning/05-Ensemble_Learning.ipynb)
