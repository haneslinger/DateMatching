import numpy as np

# keras imports 
from keras.models import Sequential
from keras.layers import Dense

def trainModel(x, y):
    # build model
    model = Sequential()
    
    model.add(Dense(units = 4,
                    activation = 'relu',
                    use_bias = True))
                    
    model.add(Dense(units = 1,
                    activation = 'relu',
                    use_bias = True))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    
    # set data into place
    model.fit(x, y,
              epochs = 5,
              verbose = 2,)


if __name__ == '__main__':
    x = np.arange(10)
    y = x * 2
    
    trainModel(x, y)
    
    print("I did it")
    
    
    