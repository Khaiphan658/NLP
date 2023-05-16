from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

# đọc dữ liệu từ file CSV:
data = pd.read_csv('C:/Users/acer/Downloads/XLNNTN/Datasets/news.csv')

# chia dữ liệu thành hai phần:
X = data['text']
y = data['label']

y = y.map({'FAKE': 0, 'REAL': 1})

# Xử lý dữ liệu:
tokenizer = Tokenizer(num_words = 5000 , filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split = ' ')
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=500)

# Chia dữ liệu để huấn luyện và xác nhận:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình LSTM Neural Network:
inputs = Input(shape=(500,))
embedding = Embedding(input_dim=5000, output_dim=32)(inputs)
lstm = LSTM(units=64)(embedding)
outputs = Dense(units=1, activation='sigmoid')(lstm)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình:
model.fit(X_train, y_train, epochs = 5, batch_size=32)

# Đánh giá mô hình:
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print("accuracy:", acc)

# Lưu mô hình
model.save('RNN.pkl')

# Chuyển đổi văn bản:
fake_news='Daniel Greenfield, a Shillman Journalism Fellow at the Freedom Center, is a New York writer focusing on radical Islam. In the final stretch of the election, Hillary Rodham Clinton has gone to war with the FBI.'
real_news='U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.'

text_seq = tokenizer.texts_to_sequences([fake_news])
text_pad = pad_sequences(text_seq, maxlen=500)

pred = model.predict(text_pad)

if pred > 0.5:
    print('\nTin thật')
else:
    print('\nTin giả')