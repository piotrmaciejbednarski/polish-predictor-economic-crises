import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

class EconomicCrisisPredictor:
    """
    Economic Crisis Predictor based on NLP analysis of economic news.
    This simplified architecture combines sentiment analysis, contextual embedding, 
    and crisis-specific lexical features to predict crisis probability.
    """
    
    def __init__(self, use_pretrained=True):
        """
        Initialize the crisis predictor model
        
        Parameters:
        -----------
        use_pretrained: bool
            Whether to use pre-trained sentence transformer for embeddings
        """
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Słownik terminów związanych z kryzysem
        from data.crisis_lexicon import crisis_lexicon
        self.crisis_lexicon = crisis_lexicon

        # Frazy niepewności ekonomicznej
        from data.uncertainty import uncertainty_phrases
        self.uncertainty_phrases = uncertainty_phrases
        
        # Initialize text vectorizer for crisis keywords
        self.crisis_vectorizer = CountVectorizer(
            vocabulary=list(self.crisis_lexicon.keys()),
            binary=False
        )
        
        # Initialize text vectorizer for uncertainty phrases
        self.uncertainty_vectorizer = CountVectorizer(
            vocabulary=[p.lower() for p in self.uncertainty_phrases],
            ngram_range=(1, 3),
            binary=True
        )
        
        # Initialize embedding method
        self.use_pretrained = use_pretrained
        if use_pretrained:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer('OrlikB/st-polish-kartonberta-base-alpha-v1')
            except:
                print("Could not load SentenceTransformer. Using TF-IDF instead.")
                self.use_pretrained = False
                self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        else:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        
        # Crisis prediction model
        self.crisis_model = None
        
    def extract_features(self, texts):
        """
        Extract features from texts for crisis prediction
        
        Parameters:
        -----------
        texts: list of str
            List of economic texts to analyze
        
        Returns:
        --------
        features: numpy.ndarray
            Feature matrix for crisis prediction
        """
        # Extract sentiment scores
        sentiment_scores = []
        for text in texts:
            scores = self.sentiment_analyzer.polarity_scores(text)
            sentiment_scores.append([scores['neg'], scores['neu'], scores['pos'], scores['compound']])
        
        sentiment_features = np.array(sentiment_scores)
        
        # Extract crisis keyword features
        if len(texts) > 0:
            # Convert to float64 to avoid type mismatch issues
            crisis_features = self.crisis_vectorizer.transform(texts).toarray().astype(np.float64)
            
            # Weight the crisis features by their lexicon values
            for i, word in enumerate(self.crisis_lexicon.keys()):
                crisis_features[:, i] *= self.crisis_lexicon[word]
                
            # Extract uncertainty phrases
            uncertainty_features = self.uncertainty_vectorizer.transform(texts).toarray()
            
            # Calculate additional statistical features
            # 1. Crisis keywords density
            text_lengths = np.array([len(text.split()) for text in texts]).reshape(-1, 1)
            keyword_density = np.sum(crisis_features, axis=1).reshape(-1, 1) / np.maximum(text_lengths, 1)
            
            # 2. Crisis keyword diversity
            keyword_presence = (crisis_features > 0).astype(int)
            keyword_diversity = np.sum(keyword_presence, axis=1).reshape(-1, 1) / len(self.crisis_lexicon)
            
            # 3. Crisis severity score (weighted average of crisis keywords)
            severity_score = np.sum(crisis_features, axis=1).reshape(-1, 1) / np.maximum(np.sum(keyword_presence, axis=1), 1).reshape(-1, 1)
            
            # 4. Uncertainty score
            uncertainty_score = np.sum(uncertainty_features, axis=1).reshape(-1, 1)
        else:
            # Handle empty list case
            crisis_features = np.empty((0, len(self.crisis_lexicon)))
            uncertainty_features = np.empty((0, len(self.uncertainty_phrases)))
            keyword_density = np.empty((0, 1))
            keyword_diversity = np.empty((0, 1))
            severity_score = np.empty((0, 1))
            uncertainty_score = np.empty((0, 1))
        
        # Text embeddings
        if self.use_pretrained:
            if len(texts) > 0:
                embeddings = self.sentence_model.encode(texts)
            else:
                embeddings = np.empty((0, self.sentence_model.get_sentence_embedding_dimension()))
        else:
            if len(texts) > 0:
                # Use TF-IDF for embeddings when pretrained model is not available
                embeddings = self.tfidf_vectorizer.fit_transform(texts).toarray()
            else:
                embeddings = np.empty((0, 0))
        
        # Combine all features
        if len(texts) > 0:
            statistical_features = np.hstack([
                keyword_density, 
                keyword_diversity, 
                severity_score,
                uncertainty_score
            ])
            
            # Concatenate all feature sets
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
        Analyze sentiment of a text
        
        Parameters:
        -----------
        text: str
            Economic text to analyze
        
        Returns:
        --------
        sentiment: dict
            Dictionary with sentiment scores
        """
        return self.sentiment_analyzer.polarity_scores(text)
    
    def fit(self, texts, crisis_labels):
        """
        Train the crisis prediction model
        
        Parameters:
        -----------
        texts: list of str
            List of economic texts
        crisis_labels: list of int
            Binary labels (1 = crisis, 0 = no crisis)
        """
        # Extract features
        X = self.extract_features(texts)
        y = np.array(crisis_labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train ensemble model
        # Using RandomForest for robustness and handling of different feature types
        self.crisis_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        
        # Train model
        self.crisis_model.fit(X_train, y_train)
        
        # Evaluate model
        train_preds = self.crisis_model.predict(X_train)
        val_preds = self.crisis_model.predict(X_val)
        train_proba = self.crisis_model.predict_proba(X_train)[:, 1]
        val_proba = self.crisis_model.predict_proba(X_val)[:, 1]
        
        print("Training accuracy:", accuracy_score(y_train, train_preds))
        print("Validation accuracy:", accuracy_score(y_val, val_preds))
        print("Training AUC-ROC:", roc_auc_score(y_train, train_proba))
        print("Validation AUC-ROC:", roc_auc_score(y_val, val_proba))
        
    def predict(self, text):
        """
        Predict crisis probability from text
        
        Parameters:
        -----------
        text: str
            Economic text to analyze
        
        Returns:
        --------
        result: dict
            Dictionary with sentiment score and crisis probability
        """
        # Analyze sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Extract features
        features = self.extract_features([text])
        
        # Predict crisis probability
        crisis_prob = self.crisis_model.predict_proba(features)[0, 1]
        
        # Calculate crisis score based on lexicon as a sanity check
        crisis_score = 0
        word_count = len(text.split())
        
        for word, weight in self.crisis_lexicon.items():
            if re.search(r'\b' + word + r'\b', text.lower()):
                crisis_score += weight
        
        crisis_score = min(1.0, crisis_score / (word_count * 0.05))
        
        # Calculate uncertainty score
        uncertainty_score = 0
        for phrase in self.uncertainty_phrases:
            if phrase.lower() in text.lower():
                uncertainty_score += 1
        
        uncertainty_score = min(1.0, uncertainty_score / len(self.uncertainty_phrases))
        
        # Get feature importance for explainability
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
