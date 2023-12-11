This is an SVM implementation with numpy and the quadratic programming framework CVXOPT.

The implementation includes an SVM module with Linear and Gaussian kernels and a test to showcase its abillity to classify data.

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
