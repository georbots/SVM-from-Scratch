This is an SVM implementation with numpy and the quadratic programming framework CVXOPT.

The implementation includes an SVM module with Linear and Gaussian kernels and a test to showcase its abillity to classify data.

# SVM
The Support Vector Machine, was firstly proposed by Vladimir Vapnik and Alexey Chervonenkis in the ealry 1960s but was not published until 1995[^1]. It is an algorithm that aims to classify data points of two discrete classes by discovering the optimal hyperplane to separate them in an n-dimensional space. 

The mathematical modelling of the problem is as follows:
The classes $y_{i}$ are defined as +1 for positive and -1 for negative. 

We need to define a weight matrix W and a bias b such as:
$$w \cdot x_{\text{positive}} + b \geq 1$$

$$w \cdot x_{\text{negative}} + b \leq -1$$


Two test cases are shown. One with linearly separable data:
![Image Alt Text](images/Linear_example_synthetic_dataset.png)

The Linear classifier identifies the support vectors in the train set:
![Image Alt Text](images/Linear_example_svm_train_solution.png)

And classifies the samples in the test set using the support vectors:
![Image Alt Text](images/Linear_example_svm_test_solution.png)

And non-linearly seaprable data:
![Image Alt Text](images/Non_linear_example_synthetic_dataset.png)

The Gaussian classifier identifies the support vectors in the train set:
![Image Alt Text](images/Non_linear_example_svm_train_solution.png)

And classifies the samples in the test set using the support vectors:
![Image Alt Text](images/Non_linear_example_svm_test_solution.png)


[^1]:Cortes, Corinna, and Vladimir Vapnik. "Support-vector networks." Machine learning 20 (1995): 273-297.
