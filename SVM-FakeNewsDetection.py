import pandas as pd
import string
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# đọc dữ liệu từ file CSV
data = pd.read_csv('C:/Users/acer/Downloads/XLNNTN/Datasets/news.csv')

# chia dữ liệu thành hai phần:
X = data['text']
y = data['label']

# loại bỏ các ký tự không cần thiết và chuyển đổi văn bản thành dạng chữ thường:
def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

# tạo bộ vector đặc trưng bằng phương pháp bag of words
vectorizer = CountVectorizer(preprocessor=preprocess, stop_words='english')
X = vectorizer.fit_transform(X)

# chia dữ liệu thành hai phần: tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# huấn luyện mô hình SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# đánh giá độ chính xác, độ tin cậy, độ phủ và F1_score của mô hình trên tập kiểm tra
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='REAL')
recall = recall_score(y_test, y_pred, pos_label='REAL')
f1score = f1_score(y_test, y_pred, pos_label='REAL')
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1_score:', f1score)

# lưu mô hình:
file='SVM.pkl'
file1='Vector.pkl'
joblib.dump(vectorizer, file1)
joblib.dump(model, file)

fake_news='Daniel Greenfield, a Shillman Journalism Fellow at the Freedom Center, is a New York writer focusing on radical Islam. In the final stretch of the election, Hillary Rodham Clinton has gone to war with the FBI.'
real_news='U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.'

text = preprocess(fake_news)
#text = preprocess(real_news)
content_vector = vectorizer.transform([text])

label = model.predict(content_vector)[0]

if label == 'REAL':
    print('\nTin thật.')
else:
    print('\nTin giả.')