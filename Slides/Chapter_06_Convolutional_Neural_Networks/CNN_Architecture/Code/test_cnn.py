conv_test_accuracy = Convolutional_model.evaluate(x_test, testY)[1]
print("Test Accuracy",np.round((conv_test_accuracy)*100,2))

---------------------------------------------------------------------------

313/313 [==============================] - 1s 3ms/step - loss: 2.2610 - accuracy: 0.6778
Test Accuracy 67.78