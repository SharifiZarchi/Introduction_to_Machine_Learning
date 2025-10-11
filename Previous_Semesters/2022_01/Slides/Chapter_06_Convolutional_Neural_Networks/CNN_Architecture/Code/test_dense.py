dense_test_accuracy = dense_model.evaluate(dense_test, testY)[1]
print("Test Accuracy",np.round((dense_test_accuracy)*100,2))

---------------------------------------------------------------------------

313/313 [==============================] - 1s 3ms/step - loss: 1.4785 - accuracy: 0.5059
Test Accuracy 50.59