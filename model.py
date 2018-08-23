import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense

class My_Model():
    def __init__(self):
        # init member varible model
        self.model = Sequential() 
        # build model
        self.model.add(Dense(input_dim  = 1,
                    output_dim = 1,
                    init = 'normal',
                    activation = 'linear',
                    use_bias = True))
        # compile model
        self.model.compile(loss='mean_squared_error',
                      optimizer='SGD',
                      metrics=['accuracy'])

    def trainModel(self, x, y):    
        # train model
        self.model.fit(x, y, epochs = 200, verbose = 1)

    def testModel(self, x):
        # test model
        return (self.model.predict(x, verbose = 1))

# main
if __name__ == '__main__':
    # create data
    x = np.linspace(-1,1,101)
    y = x*2
    
    # create and train model
    model = My_Model()
    model.trainModel(x, y)
    
    # test model
    test = np.array([0, .2, -3, 2, -5]) 
    model.testModel(x)
    
    print("I did it")
    
    
    
