from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# إعداد Flask
app = Flask(__name__)

# تحميل النموذج والمحولات
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
try:
    model.load_state_dict(torch.load("sentiment_model_bert.pth", map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    print("Error: sentiment_model_bert.pth not found. Make sure the model file is in the correct directory.")
    exit()

# تحليل المشاعر
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    sentiment = torch.argmax(outputs.logits, dim=1).item()
    return sentiment

# تغيير النبرة (هذه الدالة الآن ترجع وصفًا للنبرة فقط)
def adjust_tone(sentiment):
    if sentiment == 4:
        return "positive"
    elif sentiment == 0:
        return "negative"
    else:
        return "neutral"

# الصفحة الرئيسية
@app.route('/', methods=['GET', 'POST'])
def index():
    result = "Not Yet Analyzed"
    if request.method == 'POST':
        user_text = request.form['user-text']
        sentiment_value = analyze_sentiment(user_text)
        tone = adjust_tone(sentiment_value)
        result = f"Sentiment: {tone.capitalize()}"

    return render_template("index.html", result=result)

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True)