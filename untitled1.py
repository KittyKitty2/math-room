import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler

CHANNEL_ID = "@neironnii_kuci"

file_path = 'data.json'

data_json = pd.read_json(file_path)
df = pd.DataFrame({
    'Problem': data_json['Problem'],
    'Mathematical Expression': data_json['Mathematical Expression'],
    'Result': data_json['Result'],
})

df['Problem'] = pd.to_numeric(df['Problem'], errors='coerce').fillna(0)

X = df[['Mathematical Expression', 'Result']]
y = df['Problem']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# День 3
max_length = 6
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

# День 4
def start(update, context):
    chat_id = update.effective_chat.id
    keyboard = [[InlineKeyboardButton("Я подписался!", callback_data='check_subscription')]]
    context.bot.send_message(chat_id, "Для использования бота подпишитесь на наш канал: " + CHANNEL_ID, reply_markup=InlineKeyboardMarkup(keyboard))

def check_subscription(update, context):
    query = update.callback_query
    user_id = query.from_user.id
    try:
        chat_member = context.bot.get_chat_member(CHANNEL_ID, user_id)
        if chat_member.status in ["creator", "administrator", "member"]:
            query.edit_message_text("Спасибо за подписку! Теперь вы можете использовать бота.")
        else:
            query.answer("Вы все еще не подписаны.")
    except telegram.error.BadRequest:
        query.answer("Произошла ошибка. Попробуйте позже.")

def solve_task(update, context):
      chat_member = context.bot.get_chat_member(CHANNEL_ID, chat_id)
      if chat_member.status not in ["creator", "administrator", "member"]:
        keyboard = [[InlineKeyboardButton("Я подписался!", callback_data='check_subscription')]]
        context.bot.send_message(chat_id, "Для использования бота подпишитесь на наш канал: " + CHANNEL_ID, reply_markup=InlineKeyboardMarkup(keyboard))
        return

      if chat_member.status in ["creator", "administrator", "member"]:
        task = update.message.text

        task_df = pd.DataFrame({'Problem': [task],
                                'Mathematical Expression': [],
                                'Result': []})

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
