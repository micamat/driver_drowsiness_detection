import pandas
from keras import optimizers
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.models import model_from_json
from sklearn import model_selection
import numpy as np

def load_data_set(file_name):
    data = pandas.read_csv(file_name)
    goal = data['drowsy']
    data = data.drop('drowsy', axis=1)
    
    model = model_selection.StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=1)
    gen = model.split(data, goal)
    
    xTrain, yTrain, xTest, yTest = [], [], [], []
    for train_idx, test_idx in gen:
        xTrain = data.loc[train_idx]
        yTrain = goal.loc[train_idx]
        xTest = data.loc[test_idx]
        yTest = goal.loc[test_idx]
        
    return xTrain, yTrain, xTest, yTest

def fit(xTrain, yTrain):
    model = Sequential()
    model.add(Dense(100, input_dim=4, init="uniform", activation="relu")) # 100 je broj neurona; dodali smo input layer
    model.add(Dense(1)) # output layer, sto je zapravo klasifikator
    model.add(Activation("sigmoid"))
    
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(xTrain, yTrain, epochs=800) #bilo je epochs=500
    
    return model

def evaluate(xTest, yTest, model, file_name):
    score = model.evaluate(xTest, yTest)
    with open(file_name + ".json", "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(file_name + ".h5")

    return score

def load_model(file_name):
    jsonFile = open(file_name+".json", 'r')
    loadedModelJson = jsonFile.read()
    jsonFile.close()
    loadedModel = model_from_json(loadedModelJson)
    loadedModel.load_weights(file_name + ".h5")
    return loadedModel

def predict(data):
    model = load_model("model")
    prediction = model.predict(data)
    return prediction