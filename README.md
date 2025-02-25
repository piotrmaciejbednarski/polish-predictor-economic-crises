# Dokumentacja Projektu "Polish Predictor Economic Crises"

## Spis treści

1. [Wprowadzenie](#wprowadzenie)
2. [Architektura systemu](#architektura-systemu)
3. [Kluczowe komponenty](#kluczowe-komponenty)
4. [Techniczne aspekty rozwiązania](#techniczne-aspekty-rozwiązania)
5. [Jak działa model predykcyjny](#jak-działa-model-predykcyjny)
6. [Instrukcja użytkowania](#instrukcja-użytkowania)
7. [Propozycje ulepszeń](#propozycje-ulepszeń)

## Wprowadzenie

"Polish Predictor Economic Crises" to narzędzie do analizy i przewidywania kryzysów ekonomicznych na podstawie tekstów w języku polskim. Projekt wykorzystuje techniki przetwarzania języka naturalnego (NLP) i uczenia maszynowego do wykrywania wzorców i sygnałów ostrzegawczych w artykułach, raportach ekonomicznych i innych tekstach finansowych.

System analizuje teksty pod kątem:

- Sentymentu (pozytywny/neutralny/negatywny wydźwięk)
- Słów kluczowych związanych z kryzysem
- Fraz wskazujących na niepewność ekonomiczną

Na tej podstawie model przewiduje prawdopodobieństwo wystąpienia kryzysu gospodarczego.

## Architektura systemu

System składa się z kilku kluczowych modułów:

1. **Analizator sentymentu tekstu** (`PolishSentimentAnalyzer`) - odpowiada za ocenę wydźwięku emocjonalnego tekstu
2. **Predyktor kryzysów ekonomicznych** (`EconomicCrisisPredictor`) - główny komponent wykorzystujący różne cechy tekstu do przewidywania
3. **Słowniki specjalistyczne**:
   - Leksykon terminów kryzysowych z przypisanymi wagami
   - Frazy niepewności ekonomicznej
4. **System wizualizacji wyników** (moduł `visualize.py`)

## Kluczowe komponenty

### 1. PolishSentimentAnalyzer

Klasa opakowująca model do analizy sentymentu w języku polskim. Implementuje interfejs kompatybilny z analizatorem VADER, co ułatwia wymienność modeli. Klasa ta analizuje tekst i zwraca wyniki sentymentu w postaci wartości:

- `neg` - stopień negatywnego wydźwięku
- `neu` - stopień neutralnego wydźwięku
- `pos` - stopień pozytywnego wydźwięku
- `compound` - całościowa ocena sentymentu

### 2. EconomicCrisisPredictor

Główna klasa predykcyjna, która łączy różne źródła informacji:

- Analizę sentymentu
- Występowanie słów kluczowych związanych z kryzysem
- Występowanie fraz niepewności ekonomicznej
- Osadzanie tekstu (text embedding) przy użyciu modeli kontekstowych

Klasa ta posiada funkcje do:

- Ekstrakcji cech z tekstów (`extract_features`)
- Trenowania modelu predykcyjnego (`fit`)
- Przewidywania prawdopodobieństwa kryzysu (`predict`)

### 3. Słowniki specjalistyczne

Projekt wykorzystuje dwa główne słowniki:

- **crisis_lexicon.py** - zawiera słownik terminów związanych z kryzysem ekonomicznym wraz z przypisanymi im wagami oznaczającymi siłę powiązania (od 0.3 do 0.9)
- **uncertainty.py** - zawiera frazy wskazujące na niepewność ekonomiczną używane do wykrywania niepokojących sygnałów

## Techniczne aspekty rozwiązania

### Tokenizacja

Tokenizacja to proces dzielenia tekstu na mniejsze jednostki zwane tokenami. W projekcie wykorzystujemy tokenizatory z biblioteki NLTK:

```python
nltk.download('punkt')
```

Tokenizacja pozwala na:

- Podzielenie tekstu na zdania
- Podzielenie zdań na słowa
- Przygotowanie tekstu do dalszej analizy (np. usunięcie znaków przestankowych)

### Wektoryzacja tekstu

Wektoryzacja to proces przekształcania tekstu w formę liczbową zrozumiałą dla algorytmów uczenia maszynowego. W projekcie wykorzystywane są dwie główne metody wektoryzacji:

1. **CountVectorizer** - zamienia tekst na wektor częstości występowania słów

   ```python
   self.crisis_vectorizer = CountVectorizer(
       vocabulary=list(self.crisis_lexicon.keys()),
       binary=False
   )
   ```

2. **TfidfVectorizer** (Term Frequency-Inverse Document Frequency) - podobnie jak CountVectorizer, ale uwzględnia również wagę słowa w kontekście całego korpusu tekstów.

Wektoryzacja pozwala na identyfikację ważnych słów i fraz w tekście oraz przekształcenie ich w cechy używane przez model predykcyjny.

### Analiza sentymentu

Analiza sentymentu to technika NLP mająca na celu określenie emocjonalnego wydźwięku tekstu. W naszym projekcie wykorzystujemy model dedykowany dla języka polskiego, który:

- Analizuje tekst pod kątem pozytywnego, negatywnego i neutralnego wydźwięku
- Zwraca wartość złożoną (compound) jako ogólną ocenę sentymentu
- Pomaga wykryć niepokojące sygnały w tekstach ekonomicznych

### Random Forest

Random Forest (Las losowy) to algorytm uczenia maszynowego wykorzystywany w projekcie do przewidywania kryzysów. Jest to metoda zespołowa (ensemble method) oparta na wielu drzewach decyzyjnych:

```python
self.crisis_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```

Kluczowe parametry:

- `n_estimators=100` - liczba drzew w lesie
- `max_depth=10` - maksymalna głębokość każdego drzewa
- `class_weight='balanced'` - zrównoważenie klas (ważne przy niezbalansowanych danych)

Random Forest jest odporny na przeuczenie (overfitting) i dobrze radzi sobie z różnymi typami cech, co czyni go odpowiednim wyborem dla tego projektu.

### Metryki oceny modelu

W projekcie wykorzystujemy dwie główne metryki do oceny skuteczności modelu:

1. **Accuracy (Dokładność)** - procent poprawnie sklasyfikowanych próbek

   ```python
   accuracy_score(y_val, val_preds)
   ```

2. **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)** - miara zdolności modelu do rozróżniania między klasami (kryzys/brak kryzysu)
   ```python
   roc_auc_score(y_val, val_proba)
   ```

AUC-ROC przyjmuje wartości od 0 do 1, gdzie:

- 0.5 - model przewiduje losowo
- bliżej 1.0 - model skutecznie identyfikuje kryzysy
- bliżej 0.0 - model systematycznie się myli (można odwrócić predykcje)

## Jak działa model predykcyjny

Proces predykcji kryzysu ekonomicznego obejmuje następujące kroki:

1. **Przygotowanie danych**:

   - Wczytanie tekstów ekonomicznych
   - Opcjonalne preprocessowanie (usunięcie stopwords, lematyzacja, itp.)

2. **Ekstrakcja cech**:

   - Analiza sentymentu każdego tekstu
   - Identyfikacja terminów związanych z kryzysem
   - Wykrycie fraz niepewności ekonomicznej
   - Osadzanie tekstu przy użyciu modeli językowych

3. **Trenowanie modelu**:

   - Podział danych na zbiór treningowy i walidacyjny
   - Trenowanie klasyfikatora Random Forest
   - Ewaluacja wyników na zbiorze walidacyjnym

4. **Predykcja**:
   - Analiza nowego tekstu
   - Ekstrakcja cech
   - Przewidywanie prawdopodobieństwa kryzysu

## Instrukcja użytkowania

### Inicjalizacja modelu

```python
from predictor import EconomicCrisisPredictor

# Utworzenie instancji predyktora
predictor = EconomicCrisisPredictor(use_pretrained=True)
```

### Trenowanie modelu

```python
# Przykładowe dane treningowe
texts = ["Polska gospodarka notuje spadek PKB trzeci kwartał z rzędu...",
         "Inwestorzy z optymizmem patrzą na rynek akcji...",
         # więcej tekstów...
        ]

crisis_labels = [1, 0, ...]  # 1 - kryzys, 0 - brak kryzysu

# Trenowanie modelu
predictor.fit(texts, crisis_labels)
```

### Przewidywanie kryzysu

```python
# Analiza nowego tekstu
text = "Bank centralny podnosi stopy procentowe w odpowiedzi na rosnącą inflację..."
result = predictor.predict(text)

print(f"Prawdopodobieństwo kryzysu: {result['crisis_probability']:.2%}")
print(f"Sentyment: {result['sentiment']}")
```

### Wizualizacja wyników

```python
from visualize import create_visualization

# Analiza wielu tekstów
texts = [...]
results = [predictor.predict(text) for text in texts]

# Wizualizacja wyników
create_visualization(results, texts)
```

## Propozycje ulepszeń

1. **Ulepszenie analizy sentymentu**:

   - Dostrojenie modelu sentymentu specyficznie do tekstów ekonomicznych
   - Użycie większego i bardziej zróżnicowanego zbioru treningowego
   - Dodanie analizy kontekstowej uwzględniającej specyfikę branży finansowej

2. **Rozszerzenie leksykonów**:

   - Dodanie terminów specyficznych dla polskiej gospodarki
   - Uwzględnienie kontekstu regionalnego (UE, Europa Wschodnia)
   - Dodanie dynamicznego aktualizowania wag w słowniku kryzysowym na podstawie aktualnych danych

3. **Ulepszenie modelu predykcyjnego**:

   - Wypróbowanie innych algorytmów (np. XGBoost, głębokie sieci neuronowe)
   - Zastosowanie technik ensemble łączących różne modele
   - Implementacja analizy szeregów czasowych do wykrywania trendów

4. **Rozszerzenie funkcjonalności**:

   - Dodanie modułu do analizy danych ekonomicznych (nie tylko tekstów)
   - Implementacja automatycznego śledzenia źródeł informacji (np. RSS, portale ekonomiczne)
   - Dodanie interfejsu użytkownika (API lub dashboard)

5. **Poprawa oceny modelu**:

   - Zastosowanie cross-walidacji
   - Dodanie innych metryk oceny (precyzja, recall, F1)
   - Monitorowanie wydajności modelu w czasie

6. **Optymalizacja wydajności**:

   - Implementacja równoległego przetwarzania dla dużych zbiorów tekstów
   - Optymalizacja pamięciowa dla dużych modeli językowych
   - Dodanie mechanizmów cache'owania dla powtarzających się analiz

7. **Interpretacja modelu**:

   - Dodanie narzędzi do wyjaśniania predykcji (np. SHAP values, LIME)
   - Wizualizacja ważności cech w procesie predykcji
   - Śledzenie zmian w czasie i identyfikacja punktów zwrotnych

8. **Rozbudowanie wizualizacji**:
   - Interaktywne wykresy pokazujące trendy w czasie
   - Mapy ciepła wskazujące na najważniejsze czynniki ryzyka
   - Wizualizacje sieciowe pokazujące powiązania między różnymi aspektami kryzysu
