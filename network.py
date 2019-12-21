from keras.layers import Dense, Conv2D, Flatten

def layers():
    return [
        Conv2D(filters=32,
               kernel_size=(8, 8),
               strides=4,
               padding='valid',
               activation='relu'),
        Conv2D(filters=64,
               kernel_size=(4, 4),
               strides=2,
               padding='valid',
               activation='relu'),
        Conv2D(filters=64,
               kernel_size=(3, 3),
               strides=1,
               padding='valid',
               activation='relu'),
        Flatten(),
        Dense(units=512, activation='relu')
    ]
