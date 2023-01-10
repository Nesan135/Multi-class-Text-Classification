# Libraries
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
import os, sklearn, pickle, re
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

# Load data
DATA_PATH = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
df = pd.read_csv(DATA_PATH)

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
print(df.isna().sum())
print(df.columns)
print(df['category'].unique())
# Observations : No null data, contains duplicates (99), all data are dstring type, values to predict : ['tech' 'business' 'sport' 'entertainment' 'politics']
# view duplicates
df.duplicated().sum()
# drop duplicates
df = df.drop_duplicates()

# cleaning
for index,data in enumerate(df['text']):
    # remove tags
    data = re.sub(r'@\S*', '', data)
    # remove HTML tags
    data = re.sub(r'<.*?>','', data)
    # remove URLS
    data = re.sub(r'bit.ly?:\S*', '', data)
    data = re.sub(r'https:\S*', '', data)
    # remove special char, numbers and lower case
    data = re.sub(r'[^a-zA-Z]', ' ', data).lower()
    df['text'][index] = data

# feature selection
texts = df['text']

# determine padding length with simple statistics
median = texts.apply(lambda x: len(x.split())).median()
mean = texts.apply(lambda x: len(x.split())).mean()
max_words = texts.apply(lambda x: len(x.split())).max()
min_words = texts.apply(lambda x: len(x.split())).min()
print(f'Median: {median}\nMean: {mean}\nMax: {max_words}\nMin: {min_words}')
# we use median for padding as there are outliers

vocab_size = 10000
oov_token = '<OOV>'
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(texts)

# tokenizing
texts_tokenized = tokenizer.texts_to_sequences(texts)
# padding
texts_tokenized_padded = pad_sequences(texts_tokenized, maxlen=int(median), padding='post', truncating='post')

# encode categories
ohe = OneHotEncoder(sparse=False)
category_encoded = ohe.fit_transform(df[['category']])

# split test and train data
X_train, X_test, y_train, y_test = train_test_split(texts_tokenized_padded, category_encoded, test_size=0.2, random_state=42)

# functional API
input_layer = Input(shape=(X_train.shape[-1]))
hidden_0 = Embedding(vocab_size, 64)(input_layer)
hidden_1 = Bidirectional(LSTM(64, return_sequences=True))(hidden_0)
hidden_2 = Dropout(0.3)(hidden_1)
hidden_3 = LSTM(128)(hidden_2)
hidden_4 = Dropout(0.3)(hidden_3)
hidden_5 = Dense(128, activation='relu')(hidden_4)
hidden_6 = Dropout(0.3)(hidden_5)
output_layer = Dense(y_train.shape[-1], activation='softmax')(hidden_6)
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# callbacks
time_stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_PATH = os.path.join(os.getcwd(), 'logs', time_stamp)
tb = TensorBoard(log_dir=LOG_PATH)
es = EarlyStopping(monitor='val_loss',patience=10,verbose=1,restore_best_weights=True)

# Model training
model_hist = model.fit(X_train,y_train,epochs=160,batch_size=64,validation_data=(X_test, y_test),callbacks=[tb,es])

# visualize
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(model_hist.history['loss'])
plt.plot(model_hist.history['val_loss'])
plt.title('Loss')
plt.legend(['loss','val_loss'])
plt.subplot(1,2,2)
plt.plot(model_hist.history['acc'])
plt.plot(model_hist.history['val_acc'])
plt.title('Accuracy')
plt.legend(['acc','val_acc'])
plt.show()

# evaluate
model.evaluate(X_test, y_test)

# saving model, tokenizer and ohe labels
model.save(MSAVE_PATH)

with open(TOKENIZER_SAVE_PATH, 'wb') as file:
    pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(OHE_SAVE_PATH, 'wb') as file:
    pickle.dump(ohe, file)

# predict
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

conv_dict = {0: 'business',
             1: 'entertainment',
             2: 'politics',
             3: 'sport',
             4: 'tech'}
    
# vectorize conv(x) function
conve = np.vectorize(lambda x : conv_dict[x])

# convert y_pred into proper format for visualization
y_pred_decoded = conve(y_pred).astype(object)
y_test_decoded = ohe.inverse_transform(y_test)
y_test_decoded = y_test_decoded[:,0]
print(classification_report(y_test_decoded, y_pred_decoded))
ConfusionMatrixDisplay.from_predictions(y_test_decoded, y_pred_decoded)
plt.show()
