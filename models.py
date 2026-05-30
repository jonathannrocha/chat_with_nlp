import spacy
import random
import re
import nltk
import unicodedata

from goose3 import Goose
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect


class NLP:
    def __init__(self):
        self.sentences = {
            "en": [],
            "pt": []
        }

        self.last_response = ""

        self.random_responses = {
            "pt": [
                "Interessante.",
                "Pode explicar melhor?",
                "Curioso isso.",
                "Entendi.",
                "Continue."
            ],
            "en": [
                "Interesting.",
                "Can you explain better?",
                "Curious.",
                "I see.",
                "Continue."
            ]
        }

        self.nlp = {
            "en": spacy.load("en_core_web_lg"),
            "pt": spacy.load("pt_core_news_lg")
        }

        self.treinar(
            "https://en.wikipedia.org/wiki/Death_Note",
            "en"
        )

        self.treinar(
            "https://pt.wikipedia.org/wiki/Death_Note",
            "pt"
        )

    def clean_raw_text(self, text):
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[[a-zA-Z]+\]", "", text)
        text = re.sub(r"==.*?==", "", text)
        text = re.sub(r"http\S+|www\S+", "", text)

        text = text.lower()

        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("utf-8")

        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def treinar(self, url, lang):
        g = Goose()

        article = g.extract(url=url)

        original_text = article.cleaned_text

        nltk_lang = "portuguese" if lang == "pt" else "english"

        raw_sentences = nltk.sent_tokenize(
            original_text,
            language=nltk_lang
        )

        processed = []

        for sentence in raw_sentences:
            clean_sentence = self.clean_raw_text(sentence)

            if len(clean_sentence.split()) >= 4:
                processed.append(sentence.strip())

        self.sentences[lang].extend(processed)

    def detect_language(self, text):
        try:
            if len(text.strip()) < 4:
                return "pt"

            lang = detect(text)

            return lang if lang in ["en", "pt"] else "pt"

        except:
            return "pt"

    def welcome_message(self, text, lang):
        text = text.lower()

        if lang == "pt":
            inputs = {
                "oi",
                "olá",
                "e ai",
                "bom dia",
                "boa tarde",
                "boa noite",
                "tudo bem"
            }

            outputs = [
                "Oi",
                "Olá",
                "Tudo bem?",
                "Como posso te ajudar?",
                "Fala aí"
            ]

        else:
            inputs = {
                "hi",
                "hello",
                "hey",
                "how are you",
                "good morning",
                "good evening"
            }

            outputs = [
                "Hi",
                "Hello",
                "How can I help you?",
                "Hey there"
            ]

        for phrase in inputs:
            if phrase in text:
                return random.choice(outputs)

        return None

    def preprocessing(self, sentence, lang):
        sentence = self.clean_raw_text(sentence)

        nlp = self.nlp[lang]

        doc = nlp(sentence)

        tokens = [
            token.lemma_
            for token in doc
            if not (
                token.is_stop
                or token.is_punct
                or token.is_space
                or token.like_num
                or len(token.text) <= 1
            )
        ]

        return " ".join(tokens)

    def humor(self, text, lang):
        try:
            polarity = TextBlob(text).sentiment.polarity

            if polarity > 0.3:
                return (
                    "Parece algo positivo."
                    if lang == "pt"
                    else "Sounds positive."
                )

            elif polarity < -0.3:
                return (
                    "Isso parece meio negativo."
                    if lang == "pt"
                    else "That sounds a bit negative."
                )

            return (
                "Tom neutro."
                if lang == "pt"
                else "Neutral tone."
            )

        except:
            return ""

    def feedback_analysis(self, text, lang):
        text = self.clean_raw_text(text)

        positive_pt = {
            "sim",
            "faz sentido",
            "entendi",
            "correto",
            "ok",
            "boa",
            "certo"
        }

        negative_pt = {
            "nao",
            "errado",
            "confuso",
            "nao entendi",
            "sem sentido"
        }

        positive_en = {
            "yes",
            "correct",
            "makes sense",
            "good",
            "right",
            "understood"
        }

        negative_en = {
            "no",
            "wrong",
            "confusing",
            "doesnt make sense"
        }

        if lang == "pt":
            if any(p in text for p in positive_pt):
                return "Ótimo, então a resposta parece consistente."

            if any(n in text for n in negative_pt):
                return "Entendi. Vou tentar melhorar a resposta."

        else:
            if any(p in text for p in positive_en):
                return "Great, the answer seems consistent."

            if any(n in text for n in negative_en):
                return "Understood. I'll try to improve the answer."

        return (
            "Não consegui interpretar totalmente sua resposta."
            if lang == "pt"
            else "I could not fully understand your feedback."
        )

    def keyword_fallback(self, user_clean, sentences, lang):
        user_words = set(user_clean.split())

        matches = []

        for sentence in sentences:
            clean_sentence = self.preprocessing(sentence, lang)

            score = sum(
                1 for word in user_words
                if word in clean_sentence
            )

            if score > 0:
                matches.append((sentence, score))

        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[0][0]

        return None

    def answer(self, user_text, threshold=0.02):
        lang = self.detect_language(user_text)

        if self.last_response:
            feedback = self.feedback_analysis(user_text, lang)
            self.last_response = ""
            return feedback

        welcome = self.welcome_message(user_text, lang)

        if welcome:
            return welcome

        sentences = self.sentences[lang]

        if not sentences:
            return (
                "Desculpa, não tenho dados suficientes."
                if lang == "pt"
                else "Sorry, I don't have enough data."
            )

        cleaned_sentences = [
            self.preprocessing(sentence, lang)
            for sentence in sentences
        ]

        user_clean = self.preprocessing(user_text, lang)

        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english" if lang == "en" else None
        )

        x = tfidf.fit_transform(
            cleaned_sentences + [user_clean]
        )

        similarity = cosine_similarity(
            x[-1],
            x[:-1]
        )[0]

        best_idx = similarity.argmax()

        best_score = similarity[best_idx]

        if best_score < threshold:
            fallback = self.keyword_fallback(
                user_clean,
                sentences,
                lang
            )

            if fallback:
                response = fallback
            else:
                return random.choice(
                    self.random_responses[lang]
                )
        else:
            response = sentences[best_idx]

        mood = self.humor(response, lang)

        self.last_response = response

        if lang == "pt":
            return (
                f"{response}\n\n"
                f"{mood}\n\n"
                f"Isso faz sentido para você?"
            )

        return (
            f"{response}\n\n"
            f"{mood}\n\n"
            f"Does that make sense to you?"
        )