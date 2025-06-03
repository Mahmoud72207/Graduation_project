import pyttsx3
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import re
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# BERT model and tokenizer loading
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Data loading - still attempts to load for training/topic modeling if available
try:
    data = pd.read_csv(r'C:\Users\mohamed\Desktop\customer_feedback_data.csv')
except FileNotFoundError:
    data = pd.DataFrame(columns=['text', 'label', 'customer_feedback'])

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to existing data if available
if not data.empty and 'text' in data.columns and 'customer_feedback' in data.columns:
    data['processed_text'] = data['text'].apply(preprocess_text)
    data['processed_feedback'] = data['customer_feedback'].apply(preprocess_text)
else:
    data['processed_text'] = ""
    data['processed_feedback'] = ""

# Sentiment model training
if not data.empty and 'text' in data.columns and 'label' in data.columns:
    X = data['processed_text']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_encodings = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors="pt")
    test_encodings = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors="pt")

    train_data = torch.utils.data.TensorDataset(train_encodings['input_ids'], torch.tensor(y_train.tolist()))
    test_data = torch.utils.data.TensorDataset(test_encodings['input_ids'], torch.tensor(y_test.tolist()))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(1):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, labels = batch
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

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

    torch.save(model.state_dict(), 'sentiment_model_bert.pth')

# Topic Modeling with LDA
identified_topics = []
vectorizer = None # Initialize vectorizer to None
lda = None # Initialize lda to None

if not data.empty and 'processed_feedback' in data.columns and any(data['processed_feedback']):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(data['processed_feedback'])

    num_topics = 3
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    def display_topics(model, feature_names, no_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            topics.append(f"Topic {topic_idx + 1}: {' '.join(topic_words)}")
        return topics

    no_top_words = 5
    feature_names = vectorizer.get_feature_names_out()
    identified_topics = display_topics(lda, feature_names, no_top_words)

    topic_results = lda.transform(dtm)
    data['assigned_topic'] = topic_results.argmax(axis=1)

# Text-to-Speech (TTS) setup and sentiment-based tone adjustment
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)
voices = engine.getProperty('voices')
try:
    arabic_voice_found = False
    for voice in voices:
        if 'ar' in voice.id.lower() or any(lang.startswith('ar') for lang in voice.languages):
            engine.setProperty('voice', voice.id)
            arabic_voice_found = True
            break
    if not arabic_voice_found:
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)
        else:
            engine.setProperty('voice', voices[0].id)
except IndexError:
    engine.setProperty('voice', voices[0].id)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    sentiment_score = torch.argmax(outputs.logits, dim=1).item()
    if sentiment_score <= 1:
        return 0 # Negative
    elif sentiment_score == 2:
        return 2 # Neutral
    else:
        return 4 # Positive

def adjust_tone_based_on_sentiment(sentiment):
    if sentiment == 4:
        engine.setProperty('rate', 180)
        engine.setProperty('volume', 1.0)
        return "positive"
    elif sentiment == 0:
        engine.setProperty('rate', 120)
        engine.setProperty('volume', 0.7)
        return "negative"
    else:
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.8)
        return "neutral"

# Function to analyze a single feedback from user
def analyze_and_speak_single_feedback(feedback_text):
    processed_feedback = preprocess_text(feedback_text)
    sentiment_score = analyze_sentiment(feedback_text)
    sentiment_label = adjust_tone_based_on_sentiment(sentiment_score)

    summary_text_parts = []

    if "positive" in sentiment_label.lower():
        summary_text_parts.append("The customer feedback is positive.")
    elif "negative" in sentiment_label.lower():
        summary_text_parts.append("The customer feedback is negative.")
    else:
        summary_text_parts.append("The customer feedback is neutral.")

    # --- التعديل الرئيسي هنا: معالجة المواضيع للملاحظات الفردية ---
    if processed_feedback: # تأكد أن النص بعد المعالجة مش فاضي
        # لو الـ vectorizer و الـ lda اتدربوا على بيانات الـ CSV
        if vectorizer is not None and lda is not None and hasattr(vectorizer, 'vocabulary_'):
            try:
                # التأكد من أن النص المدخل يحتوي على كلمات موجودة في قاموس الـ vectorizer
                # وإلا سيؤدي إلى مصفوفة فارغة ومشكلة في الـ transform
                transformed_input = vectorizer.transform([processed_feedback])
                if transformed_input.sum() > 0: # لو فيه أي كلمات تم التعرف عليها
                    user_topic_results = lda.transform(transformed_input)
                    assigned_topic_idx = user_topic_results.argmax(axis=1)[0]
                    # التأكد أن الـ identified_topics مش فاضية قبل الوصول إليها
                    if identified_topics and assigned_topic_idx < len(identified_topics):
                        topic_name = identified_topics[assigned_topic_idx].replace(f'Topic {assigned_topic_idx + 1}: ', '')
                        summary_text_parts.append(f"The likely topic is: {topic_name}.")
                    else:
                        summary_text_parts.append("No specific topic could be determined from the trained model (topics not found).")
                else:
                    summary_text_parts.append("No specific topic could be determined (input words not in model vocabulary).")
            except Exception as e:
                # Catch any errors during topic transformation (e.g., if input is too different)
                summary_text_parts.append("No specific topic could be determined (topic analysis error).")
        else:
            # لو مفيش ملف CSV أو مفيش بيانات كافية لتدريب LDA، هنعمل تحليل مواضيع بسيط جداً
            try:
                # استخدام CountVectorizer بدون min_df/max_df عشان ما يعملش مشاكل مع جملة واحدة
                # max_features بتساعد في تقليل حجم القاموس لو الجملة طويلة
                temp_vectorizer = CountVectorizer(stop_words='english', max_features=5)
                temp_dtm = temp_vectorizer.fit_transform([processed_feedback])

                if temp_dtm.shape[1] > 0: # التأكد إن فيه كلمات تم استخلاصها
                    temp_lda = LatentDirichletAllocation(n_components=1, random_state=42) # موديل LDA لموضوع واحد
                    temp_lda.fit(temp_dtm)

                    temp_feature_names = temp_vectorizer.get_feature_names_out()
                    if len(temp_feature_names) > 0:
                        # عرض الكلمات الأكثر أهمية في هذا الموضوع الوحيد (أول 3 كلمات مثلاً)
                        topic_words = [temp_feature_names[i] for i in temp_lda.components_[0].argsort()[:-min(len(temp_feature_names), 3) - 1:-1]]
                        topic_name = " ".join(topic_words)
                        summary_text_parts.append(f"A possible topic is: {topic_name}.")
                    else:
                        summary_text_parts.append("No specific topic could be determined from this text.")
                else:
                    summary_text_parts.append("No specific topic could be determined from this text (no features extracted).")
            except Exception as e:
                summary_text_parts.append("No specific topic could be determined (basic topic analysis failed).")
    else:
        summary_text_parts.append("No specific topic could be determined as input text was empty or too short.")

    final_summary_text = " ".join(summary_text_parts)
    engine.say(final_summary_text)
    engine.runAndWait()


# Main execution loop
while True:
    user_feedback_input = input("Please enter customer feedback (or type 'exit' to quit): ").strip()

    if user_feedback_input.lower() == 'exit':
        final_exit_message = "Thank you for using the customer feedback analysis program. Goodbye!"
        engine.say(final_exit_message)
        engine.runAndWait()
        break
    elif user_feedback_input:
        analyze_and_speak_single_feedback(user_feedback_input)
    else:
        pass