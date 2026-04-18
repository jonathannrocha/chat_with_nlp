import spacy
import en_core_web_sm
import random
import nltk
from goose3 import Goose
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class NLP:
    def __init__(self):
        self.original_sentences = ""
        self.nlp = spacy.load('en_core_web_sm')

    def treinar( self, url): 
        nltk.download('punkt_tab')
        g=Goose()
        article=g.extract(url)
        self.original_sentences = [sentence for sentence in nltk.sent_tokenize(article.cleaned_text)]

    def welcome_message(self, text):
        welcome_words_input = ['hey','hello','hi', 'thank', 'by', 'how are you?'] 
        welcome_words_output = ['hey','hello', 'welcome', "I'm fine and you?", "Good by!"] 
        for word in text.split():
            if word.lower() in welcome_words_input: 
                return random.choice(welcome_words_output) 
            else:
                pass

    def preprocessing(self, sentence):
        sentence = sentence.lower() #tudo minúsculas

        tokens = []

        tokens = [token.text for token in self.nlp(sentence) if not (token.is_stop or token.like_num or token.is_punct or token.is_space or len(token)==1)]

        tokens=' '.join([element for element in tokens])
        return tokens

    def answer(self, user_text, threshold = 0.05):
        welcome_text = self.welcome_message( user_text)
        chatbot_answer = ''
        print( f"user { user_text}") 

        if welcome_text:
            chatbot_answer = welcome_text
            print( f"resposta:: { user_text}") 
        else:
            cleaned_sentences = []

            for sentence in self.original_sentences:
                cleaned_sentences.append(self.preprocessing(sentence))

            user_text = self.preprocessing(user_text)

            cleaned_sentences.append(user_text)

            tfidf = TfidfVectorizer()
            x_sentences = tfidf.fit_transform(cleaned_sentences)


            similarity = cosine_similarity(x_sentences[-1], x_sentences)

            sentence_index = similarity.argsort()[0][-2] #a segunda maior correspondência

            

            
            if similarity[0][sentence_index] < threshold:
                chatbot_answer += 'sorry, no answer was found'

            else:
                chatbot_answer += self.original_sentences[sentence_index]

            print( f"resposta:: { user_text}") 

        return chatbot_answer

