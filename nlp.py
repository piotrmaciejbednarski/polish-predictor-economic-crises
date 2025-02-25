import spacy
import re
from typing import List

class PolishTextPreprocessor:
    """
    Klasa do preprocessingu tekstu w języku polskim,
    zawierająca lemmatyzację i inne operacje czyszczenia tekstu przy użyciu spaCy.
    """
    def __init__(self):
        """
        Inicjalizacja preprocessora z modelem języka polskiego spaCy.
        """
        try:
            self.nlp = spacy.load('pl_core_news_lg')
            print("Załadowano preprocesor spaCy dla języka polskiego")
        except OSError:
            print("Instalowanie preprocesora spaCy dla języka polskiego...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "pl_core_news_lg"])
            self.nlp = spacy.load('pl_core_news_lg')
            print("Pomyślnie zainstalowano i załadowano spaCy")
        
    def lemmatize(self, text: str) -> str:
        """
        Lemmatyzacja tekstu w języku polskim przy użyciu spaCy.
        
        Parametry:
        -----------
        text : str
            Tekst do lemmatyzacji
            
        Zwraca:
        --------
        str
            Tekst po lemmatyzacji
        """
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])
    
    def clean_text(self, text: str) -> str:
        """
        Czyszczenie tekstu ze zbędnych znaków i formatowania.
        
        Parametry:
        -----------
        text : str
            Tekst do wyczyszczenia
            
        Zwraca:
        --------
        str
            Wyczyszczony tekst
        """
        # Usuń znaki specjalne i pozostaw tylko litery, cyfry i spacje
        text = re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s]', ' ', text)
        
        # Zamień wielokrotne spacje na pojedyncze
        text = re.sub(r'\s+', ' ', text)
        
        # Usuń spacje z początku i końca
        text = text.strip()
        
        return text
    
    def get_pos_tags(self, text: str) -> List[tuple]:
        """
        Pobieranie tagów części mowy dla tekstu.
        
        Parametry:
        -----------
        text : str
            Tekst do analizy
            
        Zwraca:
        --------
        List[tuple]
            Lista krotek (token, część_mowy)
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def get_named_entities(self, text: str) -> List[tuple]:
        """
        Rozpoznawanie jednostek nazewniczych (NER).
        
        Parametry:
        -----------
        text : str
            Tekst do analizy
            
        Zwraca:
        --------
        List[tuple]
            Lista krotek (encja, typ_encji)
        """
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def preprocess(self, text: str, lemmatize: bool = True, 
                  remove_stopwords: bool = False) -> str:
        """
        Pełny preprocessing tekstu, włączając czyszczenie i opcjonalną lemmatyzację.
        
        Parametry:
        -----------
        text : str
            Tekst do przetworzenia
        lemmatize : bool, optional
            Czy przeprowadzić lemmatyzację (domyślnie True)
        remove_stopwords : bool, optional
            Czy usunąć stopwords (domyślnie False)
            
        Zwraca:
        --------
        str
            Przetworzony tekst
        """
        # Czyszczenie tekstu
        text = self.clean_text(text)
        
        if lemmatize or remove_stopwords:
            doc = self.nlp(text)
            tokens = []
            
            for token in doc:
                if remove_stopwords and token.is_stop:
                    continue
                
                if lemmatize:
                    tokens.append(token.lemma_)
                else:
                    tokens.append(token.text)
                    
            text = ' '.join(tokens)
            
        return text
    
    def analyze_syntax(self, text: str) -> dict:
        """
        Przeprowadza szczegółową analizę składniową tekstu.
        
        Parametry:
        -----------
        text : str
            Tekst do analizy
            
        Zwraca:
        --------
        dict
            Słownik z wynikami analizy składniowej
        """
        doc = self.nlp(text)
        
        return {
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'dependencies': [(token.text, token.dep_) for token in doc],
            'named_entities': [(ent.text, ent.label_) for ent in doc.ents]
        }