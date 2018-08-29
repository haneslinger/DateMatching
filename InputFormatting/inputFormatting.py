import pickle as pkl
import numpy as np
from keras.utils import to_categorical


def fileToSeqInputfile(inputFilename, seqLen, outputFilename):
    print('begining input creation...')
    with open(inputFilename) as f:
        text = f.read()
    print('file read...')
    sequences = textToSequences(text, seqLen+1)
    Xphrases = list(map(lambda i: i[:-1], sequences))
    Yphrases = list(map(lambda i: [i[-1]], sequences))
    print('{} phrases extracted. writing to file...'.format(len(Xphrases)))
    with open(outputFilename, 'w+b') as f:
        pkl.dump(len(Xphrases), f)
        for i in range(len(Xphrases)):
            if(i % 2500 == 0 and i != 0):
                print("{} vectorized...".format(i))
            datum = [phraseToOneHot(Xphrases[i]), phraseToOneHot(Yphrases[i])]
            pkl.dump(datum, f)
    print('phrases vectorized. creation complete')

def loadInput(plkFilename):
    with open(plkFilename, 'rb') as f:
        datum = pkl.load(f)
        X = []
        Y = []
        for i in range(10000):
            if(i % 2500 == 0 and i != 0):
                print("{} vectorized...".format(i))
	    x, y =  pkl.load(f)
            X.append(x)
            Y.append(y)
    return (X, Y)

def textToSequences(text, seqLen):
    sequences = []
    for i in range(0, len(text) - seqLen):
        sequences.append(text[i:i+seqLen])
    return sequences

def phraseToOneHot(pharse):
    result =  np.empty((len(pharse), 127))
    for i, char in enumerate(pharse):
        result[i] =  to_categorical(ord(char), 127)
    return result

def oneHotToPhrase(oneHot):
    result = []
    for vector in oneHot:
	result.append(chr(np.argmax(vector)))
    return ''.join(result)

if __name__ == '__main__':
    pass
