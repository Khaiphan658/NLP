from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import string
import joblib

app = Flask(__name__, template_folder='templates')
app.config['DEBUG'] = True

# đọc dữ liệu từ file CSV:
data = pd.read_csv('C:/Users/acer/Downloads/XLNNTN/Datasets/news.csv')

# chia dữ liệu thành hai phần:
X = data['text']
y = data['label']

# SUPPORT VECTOR MACHINE:
    # Xử lý dữ liệu trong SVM:
def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text
    # tạo bộ vector đặc trưng bằng phương pháp bag of words:
vectorizer = joblib.load('Vector.pkl')
X_svm = vectorizer.fit_transform(X)
    # Chia dữ liệu để huấn luyện và xác nhận:
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y, test_size=0.2, random_state=42)
    # Load mô hình SVM:
svm_model = joblib.load('SVM.pkl')
    # Đánh giá mô hình SVM:
y_pred = svm_model.predict(X_test_svm)
acc_svm = accuracy_score(y_test_svm, y_pred)


# RECURRENT NEURAL NETWORK:
y_rnn = y.map({'FAKE': 0, 'REAL': 1})
    # loại bỏ các ký tự không cần thiết trong RNN:
tokenizer = Tokenizer(num_words=5000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=' ')
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=500)
    # Chia dữ liệu để huấn luyện và xác nhận:
X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_pad, y_rnn, test_size=0.2, random_state=42)
    # Load mô hình RNN:
rnn_model = load_model('RNN.pkl')
    # Đánh giá mô hình RNN:
loss, accuracy = rnn_model.evaluate(X_test_rnn, y_test_rnn, verbose=2)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    if request.method == "POST":
        text = request.form["text"]
        model_type = request.form["model"]

        if model_type == 'svm':
            # Preprocess the text and transform it into feature vector
            text = preprocess(text)
            text_tfidf = vectorizer.transform([text])
            
            # Predict the label using SVM model
            predicted_label = svm_model.predict(text_tfidf)[0]
            if predicted_label == 'REAL':
                result = "Tin tức thật"
            else:
                result = "Tin tức giả"
                
            return render_template("index.html", result=result, svm_acc=acc_svm)
        
        elif model_type == 'rnn':
            # Tokenize and pad the text sequence
            text_seq = tokenizer.texts_to_sequences([text])
            text_pad = pad_sequences(text_seq, maxlen=500)
            
            # Predict the label using RNN model
            pred = rnn_model.predict(text_pad)
            result = "Tin tức thật" if pred > 0.5 else "Tin tức giả"
            
            return render_template("index.html", result=result, rnn_acc=accuracy)
        return render_template("index.html")

    else:
        return "Phương thức không hợp lệ" 

if __name__ == '__main__':
    app.run(debug=True, port=5000)