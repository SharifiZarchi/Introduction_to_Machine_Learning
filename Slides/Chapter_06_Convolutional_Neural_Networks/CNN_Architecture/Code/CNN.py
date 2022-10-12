Convolutional_model = Sequential(
    [
        Conv2D(32, (3, 3), input_shape=(32, 32, 3),activation='relu', name='Conv_Layer_1'),
        MaxPool2D(pool_size=(2, 2), name='Max_Pool_1'),
        Conv2D(64, (3, 3),activation='relu', name='Conv_Layer_2'),
        Conv2D(128, (3, 3),activation='relu', name='Conv_Layer_3'),
        Flatten(name='Flatten'),
        Dense(128, activation='relu', name='Dense_Flat_1'),
        Dense(10, activation='softmax', name='Softmax_Output_Layer')
    ],
    name='Convolutional_Model'
)

Convolutional_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


Convolutional_model.summary()