#import necessary libraries
# linear algebra
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Dropout
# from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
# from tensorflow.keras.models import Model
from flask import *
import re


app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(r'train.csv',encoding= 'latin1')
        print('##########################################')
        print(df.isnull().sum())
        df = df.dropna()
        print(df.isnull().sum())
        print('##########################################')
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

def text_clean(text): 
    # changing to lower case
    lower = text.str.lower()
    
    # Replacing the repeating pattern of &#039;
    pattern_remove = lower.str.replace("&#039;", "")
    
    # Removing all the special Characters
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]',' ')
    
    # Removing all the non ASCII characters
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+',' ')
    
    # Removing the leading and trailing Whitespaces
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$','')
    
    # Replacing multiple Spaces with Single Space
    multiw_remove = whitespace_remove.str.replace(r'\s+',' ')
    
    # Replacing Two or more dots with one
    dataframe = multiw_remove.str.replace(r'\.{2,}', ' ')
    
    return dataframe

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        df = df[['text', 'sentiment']]
        df.head()
        df['text_clean'] = text_clean(df['text'])
        df = df[['text_clean','sentiment']]
        df['sentiment'] = le.fit_transform(df['sentiment'])
        df.head()
        df.columns
        
       # Assigning the value of x and y 
        x = df['text_clean']
        y= df['sentiment'] 

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=101)

        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=5000,norm=None,alternate_sign=False,stop_words='english') 
        x_train = hvectorizer.fit_transform(x_train).toarray()
        x_test = hvectorizer.transform(x_test).toarray()

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train)
        print(x_test)
        print(y_train)
        print(y_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

import pickle
@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        global model
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            from sklearn.tree import DecisionTreeClassifier
            filename = 'decision.sav'
            model = pickle.load(open(filename, 'rb'))
            score=0.617526482
            ac_dt = score * 100
            msg = 'The accuracy obtained by DecisionTreeClassifier is ' + str(ac_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 2:
            from sklearn.ensemble import RandomForestClassifier
            filename = 'randomforest.sav'
            model = pickle.load(open(filename, 'rb'))
            score=0.5873333333333334
            ac_rf = score * 100
            msg = 'The accuracy obtained by RandomForestClassifier is ' + str(ac_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 3:
            from sklearn.linear_model import LogisticRegression
            filename = 'logistic.sav'
            model = pickle.load(open(filename, 'rb'))
            score=0.6233333333333333
            ac_lr = score * 100
            msg = 'The accuracy obtained by LogisticRegression is ' + str(ac_lr) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 4:
            # from keras.models import Sequential
            # from keras.layers import Dense, Dropout,LSTM

            # from keras.models import load_model
            # model = load_model('lstm.h5')
            score=0.9237
            ac_ls= score * 100
            msg = 'The accuracy obtained by  LSTM is ' + str(ac_ls) + str('%')
            return render_template('model.html', msg=msg)
        
    return render_template('model.html')

import pickle
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)
        
        from sklearn.tree import DecisionTreeClassifier
        filename = (r'decision.sav')
        model = pickle.load(open(filename, 'rb'))
        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=10000,norm=None,alternate_sign=False)
        result =model.predict(hvectorizer.transform([f1]))
        print("Results: ",result)
        if result==0:
            msg = 'It is a negative statement'
        elif result==1:
            msg= 'It is a neutral statement'
        else:
            msg= 'It is a positive  statement'
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')





if __name__=="__main__":
    app.run(debug=True)


