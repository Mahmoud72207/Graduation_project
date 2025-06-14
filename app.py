from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.tokenize import sent_tokenize
import torch
import nltk

# تحميل punkt إذا لم يكن موجود
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

# تحميل النموذج
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

try:
    model.load_state_dict(torch.load("sentiment_model_bert.pth", map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    print("Error: sentiment_model_bert.pth not found.") # تم تغيير الرسالة للعربية
    exit()

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    sentiment = torch.argmax(outputs.logits, dim=1).item()
    return sentiment

def adjust_tone(sentiment):
    if sentiment in [3, 4]:
        return "Positive"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Negative"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_text = request.form['user-text']
        if not user_text.strip():
            return "Please enter some text to analyze." # تم تغيير الرسالة للعربية

        sentences = sent_tokenize(user_text)
        results = []
        for sentence in sentences:
            sentiment_value = analyze_sentiment(sentence)
            tone = adjust_tone(sentiment_value)
            # التعديل هنا: تغيير "المشاعر:" إلى "Sentiment:"
            results.append(f"• {sentence.strip()} ← Sentiment: {tone}")
        return "<br>".join(results)

    # تم تغيير رسالة النتيجة الافتراضية للعربية
    return render_template("index.html", result="Not yet analyzed") # تم تغيير الرسالة للعربية

if __name__ == '__main__':
    app.run(debug=True)