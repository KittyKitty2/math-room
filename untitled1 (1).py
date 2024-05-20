import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
text = pd.read_table('constant_list.txt')
tokenizer.fit_on_texts(text['CONST_pi'])
sequences = tokenizer.texts_to_sequences(text['CONST_pi'])[0]

tokenizer1 = Tokenizer()
text1 = pd.read_table('operation_list.txt')
tokenizer1.fit_on_texts(text1['add'])
sequences1 = tokenizer1.texts_to_sequences(text1['add'])[0]

seq_length = 1000
sequences = [sequences[i:i+seq_length] for i in range(0, len(sequences), seq_length)]

file_paths = ['challenge_test.json', 'dev.json', 'test.json', 'train.json']

def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

dfs = []
for file_path in file_paths:
    data_json = pd.read_json(file_path)
    df = pd.DataFrame({
        'Problem': data_json['Problem'],
        'category': data_json['category'],
        'Rationale': data_json['Rationale'],
        'options': data_json['options'],
        'correct': data_json['correct'],
        'annotated_formula': data_json['annotated_formula'],
        'linear_formula': data_json['linear_formula']
    })

    label_encoder = LabelEncoder()
    for col in ['Rationale', 'options', 'correct', 'annotated_formula', 'linear_formula']:
        df[col] = label_encoder.fit_transform(df[col])

    df['Problem'] = pd.to_numeric(df['Problem'], errors='coerce').fillna(0)
    dfs.append(df)

df_combined = pd.concat(dfs)

X = df_combined.drop(columns=['category']).values
y = pd.get_dummies(df_combined['category']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_length = 6
X_train = pad_sequences(X_train, maxlen=max_length, dtype='float32', padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, dtype='float32', padding='post', truncating='post')

vocab_size = 100000

input_layer = Input(shape=(max_length,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=10000, input_length=max_length)(input_layer)
dense_layer1 = Dense(256, activation='relu')(embedding_layer)
dense_layer2 = Dense(512, activation='relu')(dense_layer1)
dense_layer3 = Dense(1024, activation='relu')(dense_layer2)
lstm_layer = LSTM(2048, return_sequences=False)(dense_layer3)
dropout_layer = Dropout(0.6)(lstm_layer)
dense_layer4 = Dense(1024, activation='relu')(dropout_layer)
dense_layer5 = Dense(512, activation='relu')(dense_layer4)
dense_layer6 = Dense(256, activation='relu')(dense_layer5)
output_layer = Dense(y.shape[1], activation='softmax')(dense_layer6)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, batch_size=152)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=3)
print("Точность на тестовых данных: ", test_acc)
global_label_encoder = label_encoder

def start(update, context):
    update.message.reply_text('Привет! Отправьте мне задачу для решения.')

def solve_task(update, context):
    task = update.message.text

    task_df = pd.DataFrame({'Problem': [task],
                            'Rationale': [''],
                            'options': [''],
                            'correct': [''],
                            'annotated_formula': [''],
                            'linear_formula': ['']})

    task_df['Problem'] = task_df['Problem'].str.lower()

    task_df['Problem'] = pd.to_numeric(task_df['Problem'], errors='coerce').fillna(0)
    preprocessed_task = pad_sequences([task_df['Problem'].values], maxlen=max_length, dtype='float32', padding='post', truncating='post')
    prediction = model.predict(preprocessed_task)

    predicted_category = np.argmax(prediction, axis=1)[0]
    formatted_prediction = label_encoder.inverse_transform([predicted_category])[0]

    update.message.reply_text(f'Ответ: {formatted_prediction}')

def main():
    updater = Updater(token='7057782957:AAFBsUuP-tEYsUE_uduYwo_71ai7YRWjwWU')
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, solve_task))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
