# import libraries
import pandas as pd
import tensorflow, unicodedata, re, contractions, string, spacy, time, textwrap, os, datetime, pickle, json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional
from tensorflow.keras import Sequential
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

# load data
df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')

# create folder to save in
if not os.path.exists('saved_models'):
   os.makedirs('saved_models')

# define save paths to new folder
MSAVE_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')
BW_PATH = os.path.join(os.getcwd(), 'saved_models', 'best_weights.h5')
TOKENIZER_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','tokenizer.json')
OHE_SAVE_PATH = os.path.join(os.getcwd(),'saved_models','ohe.pkl')

# EDA
print(df.info())
print(df.head(5))
print(df.duplicated().sum())
print(df.isna().sum())
print(df.columns)
print(df['category'].unique())
df.drop_duplicates()

# Observations : No null data, contains duplicates (99), all data are dstring type, values to predict : ['tech' 'business' 'sport' 'entertainment' 'politics']

# Cleaning helper functions
def expand_contractions(text):
    expanded_words = [] 
    for word in text.split():
       expanded_words.append(contractions.fix(word)) 
    return ' '.join(expanded_words)

def lemmatize(text, nlp):
   doc = nlp(text)
   lemmatized_text = []
   for token in doc:
     lemmatized_text.append(token.lemma_)
   return ' '.join(lemmatized_text)

def remove_stopwords(text,nlp):          
    filtered_sentence = [] 
    doc = nlp(text)
    for token in doc:        
        if token.is_stop == False: 
          filtered_sentence.append(token.text)   
    return ' '.join(filtered_sentence)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
fake_commas = [['“','"'],['”','"'],['‘',"'"],['’',"'"]]
start = time.time()
counter = 0

for index,data in enumerate(df['text']):
    # Standardizing Accent Characters
    data = unicodedata.normalize('NFKD', data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # remove false commas
    for fake_comma in fake_commas:
        data = re.sub(fake_comma[0],fake_comma[1],data)    
    # remove tags
    data = re.sub(r'@\S*', '', data)
    # remove HTML tags
    data = re.sub('<.*?>','', data)
    # remove URLS
    data = re.sub(r'bit.ly?:\S*', '', data)
    data = re.sub(r'https:\S*', '', data)
    # remove words in boxes
    data = re.sub(r'\[.*?\]', '', data)
    # remove special char, numbers and lower case
    data = re.sub(r'[^a-zA-z.,!?/:;\"\'\s]', ' ', data).lower()
    # expand contractions
    data = expand_contractions(data)
    # remove punctuation
    data = ''.join([c for c in data if c not in string.punctuation])
    data = lemmatize(data,nlp)
    data = remove_stopwords(data,nlp)
    # #to check :
    counter +=1
    if counter%1000 == 0:
        end = time.time()
        print(counter,end-start)
        start = end
    # commit to dataframe
    df['text'][index] = data

# feature selection
review = df['text']
sentiment = df['category']

# unique number of words in all sentences
num_words = 10000
# out of vocab
oov_token = '<OOV>'
tokenizer = Tokenizer(num_words=num_words,oov_token=oov_token)
tokenizer.fit_on_texts(review)
# word_index = tokenizer.word_index
# print(dict(list(word_index.items())[0:10]))

# preprocessing
review = tokenizer.texts_to_sequences(review)

# Checking Median, mode, mean, max, min
temp = [len(review[i]) for i in range(len(review))]
median = np.median(temp)
mean = np.mean(temp)
max_words = np.max(temp)
min_words = np.min(temp)
print(f'Median: {median}\nMean: {mean}\nMax: {max_words}\nMin: {min_words}')

# we use median for padding as the data is skewed
padded_review = pad_sequences(review, maxlen=int(median), padding='post', truncating='post')

ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(sentiment[::,None])

padded_review = np.expand_dims(padded_review, axis=-1)

X_train,X_test,y_train,y_test = train_test_split(padded_review,sentiment,test_size=0.3,random_state=123)

model = Sequential()
model.add(Embedding(num_words, 64))
model.add(Bidirectional(LSTM(16, return_sequences = True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(16)))
model.add(Dense(5, activation = 'softmax'))
model.summary()
#plot_model(model,show_shapes=True,show_layer_names=True)
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['acc'])

log_path = os.path.join('log_dir',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = TensorBoard(log_dir=log_path)
es_callback = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)
model_callback = ModelCheckpoint(BW_PATH, monitor='val_loss', save_best_only='True', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001,verbose=1)
hist = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=40,batch_size=64, callbacks=[es_callback,tb_callback,model_callback,reduce_lr])
# result visualization
print('hist keys :',hist.history.keys())

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['loss','val_loss'])
plt.subplot(1,2,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Accuracy')
plt.legend(['acc','val_acc'])
plt.show()

y_tested = np.argmax(y_test, axis=1)
y_predicted = model.predict(X_test)
y_predicted = np.argmax(y_predicted, axis=1)
print(classification_report(y_tested, y_predicted))
ConfusionMatrixDisplay.from_predictions(y_tested, y_predicted)
plt.show()

# model saving
model.save(MSAVE_PATH)
with open(OHE_SAVE_PATH, 'wb') as f:
    pickle.dump(ohe,f)

token_json = tokenizer.to_json()
with open(TOKENIZER_SAVE_PATH, 'w') as f:
    json.dump(token_json,f)
