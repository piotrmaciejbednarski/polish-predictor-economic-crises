import numpy as np
import re
import nltk
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Pobierz niezbędne zasoby NLTK
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class PolishSentimentAnalyzer:
    """
    Klasa opakowująca model HuggingFace do analizy sentymentu w języku polskim,
    implementująca interfejs zgodny z VADER dla ułatwienia wymiany.
    """
    def __init__(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Używamy modelu dla języka polskiego przeznaczonego do analizy sentymentu
            model_name = "Voicelab/herbert-base-cased-sentiment"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()  # Ustawiamy model w trybie ewaluacji
            self.is_huggingface = True
            print("Załadowano polski model analizy sentymentu")
        except Exception as e:
            # Zachowanie VADER jako fallback
            print(f"Nie udało się załadować polskiego modelu sentymentu: {e}")
            print("Używanie VADER jako zapasowego analizatora (uwaga: wyniki dla języka polskiego mogą być niedokładne)")
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
            self.is_huggingface = False
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon')
    
    def polarity_scores(self, text):
        """
        Analiza sentymentu tekstu, zachowanie kompatybilne z VADER.
        
        Parametry:
        -----------
        text: str
            Tekst do analizy
            
        Zwraca:
        --------
        sentiment: dict
            Słownik z wynikami sentymentu (neg, neu, pos, compound)
        """
        if self.is_huggingface:
            # Dla modelu HuggingFace
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Polskie modele zwykle mają klasy w kolejności: [negatywny, neutralny, pozytywny]
            neg, neu, pos = scores[0].tolist()
            
            # Obliczanie wartości compound podobnie jak w VADER
            # Wartość od -1 (całkowicie negatywna) do 1 (całkowicie pozytywna)
            compound = pos - neg
            
            return {
                'neg': neg,
                'neu': neu,
                'pos': pos,
                'compound': compound
            }
        else:
            # Dla VADER
            return self.analyzer.polarity_scores(text)

class EconomicCrisisPredictor:
    """
    Predyktor Kryzysu Ekonomicznego oparty na analizie NLP wiadomości ekonomicznych.
    Ta uproszczona architektura łączy analizę sentymentu, osadzanie kontekstowe 
    oraz specyficzne dla kryzysu cechy leksykalne w celu przewidywania prawdopodobieństwa kryzysu.
    """
    
    def __init__(self, use_pretrained=True):
        """
        Inicjalizacja modelu predykcji kryzysu
        
        Parametry:
        -----------
        use_pretrained: bool
            Czy używać wstępnie wytrenowanego transformera zdań do osadzania
        """
        # Inicjalizacja analizatora sentymentu
        self.sentiment_analyzer = PolishSentimentAnalyzer()
        
        # Słownik terminów związanych z kryzysem
        from data.crisis_lexicon import crisis_lexicon
        self.crisis_lexicon = crisis_lexicon

        # Frazy niepewności ekonomicznej
        from data.uncertainty import uncertainty_phrases
        self.uncertainty_phrases = uncertainty_phrases
        
        # Inicjalizacja wektoryzatora tekstu dla słów kluczowych kryzysu
        self.crisis_vectorizer = CountVectorizer(
            vocabulary=list(self.crisis_lexicon.keys()),
            binary=False
        )
        
        # Inicjalizacja wektoryzatora tekstu dla fraz niepewności
        self.uncertainty_vectorizer = CountVectorizer(
            vocabulary=[p.lower() for p in self.uncertainty_phrases],
            ngram_range=(1, 3),
            binary=True
        )
        
        # Inicjalizacja metody osadzania tekstu
        self.use_pretrained = use_pretrained
        if use_pretrained:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer('OrlikB/st-polish-kartonberta-base-alpha-v1')
            except Exception as e:
                print(f"Nie udało się załadować SentenceTransformer: {e}")
                print("Używanie TF-IDF zamiast tego.")
                self.use_pretrained = False
                self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        else:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        
        # Model predykcji kryzysu
        self.crisis_model = None
        
    def extract_features(self, texts):
        """
        Ekstrakcja cech z tekstów do predykcji kryzysu
        
        Parametry:
        -----------
        texts: lista str
            Lista tekstów ekonomicznych do analizy
        
        Zwraca:
        --------
        features: numpy.ndarray
            Macierz cech do predykcji kryzysu
        """
        # Ekstrakcja wyników sentymentu
        sentiment_scores = []
        for text in texts:
            scores = self.sentiment_analyzer.polarity_scores(text)
            sentiment_scores.append([scores['neg'], scores['neu'], scores['pos'], scores['compound']])
        
        sentiment_features = np.array(sentiment_scores)
        
        # Ekstrakcja cech słów kluczowych kryzysu
        if len(texts) > 0:
            # Konwersja do float64, aby uniknąć problemów z niezgodnością typów
            crisis_features = self.crisis_vectorizer.transform(texts).toarray().astype(np.float64)
            
            # Ważenie cech kryzysowych wartościami z leksykonu
            for i, word in enumerate(self.crisis_lexicon.keys()):
                crisis_features[:, i] *= self.crisis_lexicon[word]
                
            # Ekstrakcja fraz niepewności
            uncertainty_features = self.uncertainty_vectorizer.transform(texts).toarray()
            
            # Obliczanie dodatkowych cech statystycznych
            # 1. Gęstość słów kluczowych kryzysu
            text_lengths = np.array([len(text.split()) for text in texts]).reshape(-1, 1)
            keyword_density = np.sum(crisis_features, axis=1).reshape(-1, 1) / np.maximum(text_lengths, 1)
            
            # 2. Różnorodność słów kluczowych kryzysu
            keyword_presence = (crisis_features > 0).astype(int)
            keyword_diversity = np.sum(keyword_presence, axis=1).reshape(-1, 1) / len(self.crisis_lexicon)
            
            # 3. Wynik nasilenia kryzysu (ważona średnia słów kluczowych kryzysu)
            severity_score = np.sum(crisis_features, axis=1).reshape(-1, 1) / np.maximum(np.sum(keyword_presence, axis=1), 1).reshape(-1, 1)
            
            # 4. Wynik niepewności
            uncertainty_score = np.sum(uncertainty_features, axis=1).reshape(-1, 1)
        else:
            # Obsługa przypadku pustej listy
            crisis_features = np.empty((0, len(self.crisis_lexicon)))
            uncertainty_features = np.empty((0, len(self.uncertainty_phrases)))
            keyword_density = np.empty((0, 1))
            keyword_diversity = np.empty((0, 1))
            severity_score = np.empty((0, 1))
            uncertainty_score = np.empty((0, 1))
        
        # Osadzanie tekstu
        if self.use_pretrained:
            if len(texts) > 0:
                embeddings = self.sentence_model.encode(texts)
            else:
                embeddings = np.empty((0, self.sentence_model.get_sentence_embedding_dimension()))
        else:
            if len(texts) > 0:
                # Używanie TF-IDF do osadzania, gdy model wstępnie wytrenowany nie jest dostępny
                embeddings = self.tfidf_vectorizer.fit_transform(texts).toarray()
            else:
                embeddings = np.empty((0, 0))
        
        # Łączenie wszystkich cech
        if len(texts) > 0:
            statistical_features = np.hstack([
                keyword_density, 
                keyword_diversity, 
                severity_score,
                uncertainty_score
            ])
            
            # Konkatenacja wszystkich zestawów cech
            features = np.hstack([
                sentiment_features,
                crisis_features,
                uncertainty_features,
                statistical_features
            ])
            
            if self.use_pretrained:
                features = np.hstack([features, embeddings])
        else:
            features = np.empty((0, 0))
            
        return features
    
    def analyze_sentiment(self, text):
        """
        Analiza sentymentu tekstu
        
        Parametry:
        -----------
        text: str
            Tekst ekonomiczny do analizy
        
        Zwraca:
        --------
        sentiment: dict
            Słownik z wynikami sentymentu
        """
        return self.sentiment_analyzer.polarity_scores(text)
    
    def fit(self, texts, crisis_labels):
        """
        Trenowanie modelu predykcji kryzysu
        
        Parametry:
        -----------
        texts: lista str
            Lista tekstów ekonomicznych
        crisis_labels: lista int
            Etykiety binarne (1 = kryzys, 0 = brak kryzysu)
        """
        # Ekstrakcja cech
        X = self.extract_features(texts)
        y = np.array(crisis_labels)
        
        # Podział danych
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Trenowanie modelu zespołowego
        # Używanie RandomForest dla odporności i obsługi różnych typów cech
        self.crisis_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        
        # Trenowanie modelu
        self.crisis_model.fit(X_train, y_train)
        
        # Ewaluacja modelu
        train_preds = self.crisis_model.predict(X_train)
        val_preds = self.crisis_model.predict(X_val)
        train_proba = self.crisis_model.predict_proba(X_train)[:, 1]
        val_proba = self.crisis_model.predict_proba(X_val)[:, 1]
        
        print("Dokładność treningowa:", accuracy_score(y_train, train_preds))
        print("Dokładność walidacyjna:", accuracy_score(y_val, val_preds))
        print("AUC-ROC treningowy:", roc_auc_score(y_train, train_proba))
        print("AUC-ROC walidacyjny:", roc_auc_score(y_val, val_proba))
        
    def predict(self, text):
        """
        Przewidywanie prawdopodobieństwa kryzysu na podstawie tekstu
        
        Parametry:
        -----------
        text: str
            Tekst ekonomiczny do analizy
        
        Zwraca:
        --------
        result: dict
            Słownik z wynikiem sentymentu i prawdopodobieństwem kryzysu
        """
        # Analiza sentymentu
        sentiment = self.analyze_sentiment(text)
        
        # Ekstrakcja cech
        features = self.extract_features([text])
        
        # Przewidywanie prawdopodobieństwa kryzysu
        crisis_prob = self.crisis_model.predict_proba(features)[0, 1]
        
        # Obliczanie wyniku kryzysu na podstawie leksykonu jako kontrola sanity
        crisis_score = 0
        word_count = len(text.split())
        
        for word, weight in self.crisis_lexicon.items():
            if re.search(r'\b' + word + r'\b', text.lower()):
                crisis_score += weight
        
        crisis_score = min(1.0, crisis_score / (word_count * 0.05))
        
        # Obliczanie wyniku niepewności
        uncertainty_score = 0
        for phrase in self.uncertainty_phrases:
            if phrase.lower() in text.lower():
                uncertainty_score += 1
        
        uncertainty_score = min(1.0, uncertainty_score / len(self.uncertainty_phrases))
        
        # Pobieranie ważności cech dla wyjaśnialności
        crisis_keywords_present = []
        for word in self.crisis_lexicon:
            if re.search(r'\b' + word + r'\b', text.lower()):
                crisis_keywords_present.append(word)
                
        uncertainty_phrases_present = []
        for phrase in self.uncertainty_phrases:
            if phrase.lower() in text.lower():
                uncertainty_phrases_present.append(phrase)
        
        return {
            'sentiment': sentiment,
            'sentiment_label': 'negative' if sentiment['compound'] <= -0.05 else 'positive' if sentiment['compound'] >= 0.05 else 'neutral',
            'crisis_probability': float(crisis_prob),
            'crisis_score': float(crisis_score),
            'uncertainty_score': float(uncertainty_score),
            'crisis_keywords': crisis_keywords_present,
            'uncertainty_phrases': uncertainty_phrases_present
        }