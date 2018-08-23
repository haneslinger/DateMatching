import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def trainModel(x, y):
    # build model
    model = Sequential()    
    model.add(Dense(input_dim  = 1,
                    output_dim = 1,
                    init = 'normal',
                    activation = 'linear',
                    use_bias = True))

    # compile model
    model.compile(loss='mean_squared_error',
                  optimizer='SGD',
                  metrics=['accuracy'])
    
    # train model
    model.fit(x, y, epochs = 200, verbose = 1)

    # test model
    test =  np.array([0, .2, -3, 2, -5])
    print(model.predict(test, verbose = 1))

# main
if __name__ == '__main__':
    x = np.linspace(-1,1,101)
    y = x*2
    
    trainModel(x, y)
    
    print("I did it")
    
    
    
