import pyttsx3
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import re

# تحميل النموذج والمحول BERT
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# تحميل البيانات
data = pd.read_csv(r'D:\Projects\Graduation_project\sentiment_data.csv')

# معالجة النصوص: تنظيف البيانات
def preprocess_text(text):
    text = text.lower()  # تحويل النص إلى حروف صغيرة
    text = re.sub(r'\d+', '', text)  # إزالة الأرقام
    text = re.sub(r'[^\w\s]', '', text)  # إزالة الترقيم
    stop_words = set(stopwords.words('english'))  # الحصول على الكلمات الشائعة
    text = ' '.join([word for word in text.split() if word not in stop_words])  # إزالة الكلمات الشائعة
    return text

# تطبيق معالجة النصوص على البيانات
data['processed_text'] = data['text'].apply(preprocess_text)

# تقسيم البيانات إلى مدخلات (X) وأهداف (y)
X = data['processed_text']
y = data['label']

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ترميز النصوص باستخدام BERT Tokenizer
def encode_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# تطبيق الترميز على البيانات
train_encodings = encode_texts(X_train.tolist())
test_encodings = encode_texts(X_test.tolist())

# إنشاء مجموعات البيانات لتدريب واختبار الموديل
train_data = torch.utils.data.TensorDataset(train_encodings['input_ids'], torch.tensor(y_train.tolist()))
test_data = torch.utils.data.TensorDataset(test_encodings['input_ids'], torch.tensor(y_test.tolist()))

# تحميل البيانات إلى DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

# استخدام المحسن (Optimizer)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# تدريب النموذج
model.train()
for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, labels = batch
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} finished, loss: {loss.item()}")

# تقييم النموذج
model.eval()
predictions = []
labels = []
for batch in test_loader:
    input_ids, label = batch
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)
    predictions.extend(preds.cpu().numpy())
    labels.extend(label.cpu().numpy())

# طباعة تقرير التصنيف
print(classification_report(labels, predictions))

# حفظ النموذج المدرب
torch.save(model.state_dict(), 'sentiment_model_bert.pth')

# إعداد محرك النص إلى كلام (TTS)
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # ضبط سرعة الصوت
engine.setProperty('volume', 1)  # ضبط حجم الصوت
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # استخدام الصوت الثاني

# تحليل المشاعر
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    sentiment = torch.argmax(outputs.logits, dim=1).item()
    return sentiment

# تعديل الصوت بناءً على المشاعر
def adjust_tone_based_on_sentiment(sentiment):
    if sentiment == 4:  # إذا كانت المشاعر إيجابية
        engine.setProperty('rate', 180)  # زيادة السرعة
        engine.setProperty('volume', 1)  # زيادة الحجم
        return "positive"
    elif sentiment == 0:  # إذا كانت المشاعر سلبية
        engine.setProperty('rate', 120)  # تقليل السرعة
        engine.setProperty('volume', 0.7)  # تقليل الحجم
        return "negative"
    else:  # إذا كانت المشاعر محايدة
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.8)
        return "neutral"

# تحويل النص إلى كلام مع تعديل الصوت بناءً على المشاعر
def text_to_speech_with_sentiment(text):
    sentences = text.split('. ')  # تقسيم النص إلى جمل
    for sentence in sentences:
        sentiment = analyze_sentiment(sentence)
        sentiment_label = adjust_tone_based_on_sentiment(sentiment)
        print(f"Sentence: {sentence}\nSentiment: {sentiment_label}")
        engine.say(sentence)
        engine.runAndWait()

# إدخال النص لتحويله إلى كلام
text = input("Enter the text you want to convert to speech: ")
text_to_speech_with_sentiment(text)
