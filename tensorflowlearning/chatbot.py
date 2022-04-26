import tensorflow as tf
import os
#want to use CPU have to uncomment bellow to disable GPU
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    # Disable all GPUS
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from http.client import responses
import json
import random
import re
import nltk
import pickle
from keras.models import model_from_yaml, model_from_json, Sequential
from keras.layers import Dense
from nltk.stem import LancasterStemmer
import numpy


nltk.download('punkt')
#https://www.youtube.com/watch?reload=9&v=NkfxjRKSnpU
#nltk.download()

stemmer= LancasterStemmer()

print("LancasterStemmer")

with open("chatbot.json") as file:
    data = json.load(file)

try:
    with open("1chatbot.pickle") as file:
        words, labels, training, output = pickle.load(file)
    
except Exception as ex:
    print(ex)
    words = []
    labels = []
    docs_x=[]
    docs_y=[]
    
    for intent in data["intents"]:
        tag= intent["tag"]
        for pattern in intent["patterns"]:
            wds= nltk.word_tokenize(pattern)
            words.extend(wds)
            docs_x.append(wds)
            docs_y.append(tag)
        
        if tag not in labels:
            labels.append(tag)
    
    words=[stemmer.stem(w.lower()) for w in words if w!='?']
    words= sorted(list(set(words)))
    
    labels= sorted(labels)
    
    training=[]
    output=[]
    
    output_empty=[0 for _ in range(len(labels)) ]
    
    for x,doc in enumerate(docs_x):
        bag=[]
        wrds=[stemmer.stem(w.lower()) for w in doc]
        
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = output_empty[:]
        output_row[labels.index(docs_y[x])]=1
        
        training.append(bag)
        output.append(output_row)
        
    training= numpy.array(training)
    output= numpy.array(output)
        
    with open("chatbot.pickle","wb") as file:
        pickle.dump((words,labels, training, output),file)

print(words)
print(labels)
print(training)
print(output)

#exit(0)

try:
    yaml_file= open("1chatbotmodel.json","r")
    loaded_model_yaml= yaml_file.read()
    yaml_file.close()
    myChatModel = model_from_json(loaded_model_yaml)
    myChatModel.load_weights("chatbotmodel.h5")
    
except Exception as exerr:
    myChatModel=Sequential()
    myChatModel.add(Dense(8, input_shape=[len(words)], activation= "relu"))
    myChatModel.add(Dense(len(labels), activation="softmax"))
    
    myChatModel.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print(training)
    print(output)
    
    myChatModel.fit(training, output, epochs=1000, batch_size=8)
    
    model_yaml= myChatModel.to_json()
    with open("chatbotmodel.json","w") as y_file:
        y_file.write(model_yaml)
        
    myChatModel.save_weights("chatbotmodel.h5")


def bag_of_words(s, words):
    bag=[0 for _ in range(len(words))]
    
    s_words= nltk.word_tokenize(s)
    s_words= [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=1
                
    return numpy.array(bag)


def predict(s):
    inputText=numpy.array([bag_of_words(s, words)])
    
    result= myChatModel.predict(inputText[0:1])
    result_index= numpy.argmax(result)
    
    tag= labels[result_index]
    
    if(result[0][result_index]>0):
        for tg in data["intents"]:
            if tg["tag"]== tag:
                responses= tg["responses"]
        
        return random.choice(responses)
        
    else:
        return None
print("predict")
val=predict("xin chao du du")
print(val)
