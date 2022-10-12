dense_model = Sequential(
    [
        Dense(2048, input_dim=32*32*3, activation='relu', name='Dense_Layer_1'),
        Dense(1024, activation='relu', name='Dense_Layer_2'),
        Dense(512, activation='relu', name='Dense_Layer_3'),
        Dense(128, activation='relu', name='Dense_Layer_4'),
        Dense(10, activation='softmax', name='Softmax_Output_Layer'),
    ],
    name='Dense_Model'
)
dense_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
dense_model.summary()