Projekt Polish Predictor of Economic Crises (PPoEC) to narzędzie do analizy tekstów ekonomicznych w języku polskim i przewidywania prawdopodobieństwa wystąpienia kryzysu gospodarczego. System wykorzystuje techniki przetwarzania języka naturalnego (NLP) i uczenia maszynowego do oceny ryzyka kryzysu na podstawie artykułów, raportów lub innych tekstów ekonomicznych.

## Struktura projektu

Projekt składa się z następujących plików:

- main.py: Punkt wejściowy do programu
- predictor.py: Główna klasa predyktora kryzysów ekonomicznych
- dataset.py: Tworzenie przykładowego zbioru danych do treningu
- visualize.py: Funkcje do wizualizacji wyników
- crisis_lexicon.py: Słownik terminów związanych z kryzysem wraz z wagami
- uncertainty.py: Lista fraz wskazujących na niepewność gospodarczą

## Jak działa system? - Step by step

### 1. Przygotowanie danych

System wykorzystuje przykładowy zbiór danych utworzony w dataset.py, który zawiera 30 tekstów ekonomicznych podzielonych na:

- 10 wyraźnych przykładów kryzysu (etykieta 1)
- 10 przykładów umiarkowanych obaw (etykieta 0)
- 10 przykładów neutralnych/pozytywnych (etykieta 0)

Teksty te służą do treningu modelu.

```python
# Przykład tworzenia zbioru danych
texts, labels = create_sample_dataset()
```

### 2. Inicjalizacja modelu

Predyktor inicjalizowany jest z określonymi parametrami, w tym czy ma korzystać z pretrainowanego modelu języka:

```python
predictor = EconomicCrisisPredictor(use_pretrained=True)
```

### 3. Ekstrakcja cech

System wyodrębnia z tekstów następujące cechy:

#### a) Cechy sentymentu

Wykorzystując analizator sentymentu VADER, system ekstrahuje oceny sentymentu (negatywny, neutralny, pozytywny, złożony).

#### b) Słowa kluczowe związane z kryzysem

System wykorzystuje zdefiniowany w crisis_lexicon.py słownik terminów kryzysowych wraz z ich wagami (od 0 do 1), np.:

- 'kryzys': 0.9
- 'recesja': 0.8
- 'bankructwo': 0.9

#### c) Frazy niepewności

System wykrywa frazy wskazujące na niepewność gospodarczą zdefiniowane w uncertainty.py, np.:

- 'niepewne perspektywy'
- 'rosnące obawy'
- 'niejasna przyszłość'

#### d) Cechy statystyczne

- Gęstość słów kluczowych (suma wag / długość tekstu)
- Różnorodność słów kluczowych (unikalne słowa / wszystkie możliwe słowa)
- Wynik dotkliwości kryzysu (średnia ważona słów kluczowych)
- Wynik niepewności (ilość wykrytych fraz niepewności)

#### e) Osadzenia tekstu (embeddings)

System używa jednego z dwóch podejść:

- Pretrainowany model BERT dla języka polskiego (`OrlikB/st-polish-kartonberta-base-alpha-v1`)
- Wektoryzacja TF-IDF jako rozwiązanie zapasowe

### 4. Trening modelu

System trenuje klasyfikator Random Forest na wyodrębnionych cechach:

```python
predictor.fit(texts, labels)
```

80% danych używane jest do treningu, a 20% do walidacji. System wyświetla metryki jakości:

- Dokładność na zbiorze treningowym
- Dokładność na zbiorze walidacyjnym
- AUC-ROC na zbiorze treningowym
- AUC-ROC na zbiorze walidacyjnym

### 5. Przewidywanie kryzysu

Wytrenowany model jest używany do przewidywania prawdopodobieństwa kryzysu dla nowych tekstów:

```python
result = predictor.predict(text)
```

Dla każdego tekstu system:

1. Analizuje sentyment
2. Ekstrahuje cechy
3. Przewiduje prawdopodobieństwo kryzysu
4. Oblicza wynik kryzysu oparty na leksykonie (jako dodatkowe potwierdzenie)
5. Oblicza wynik niepewności
6. Tworzy listę znalezionych słów kluczowych i fraz niepewności

Wynik zawiera:

- Ocenę sentymentu (negatywny/neutralny/pozytywny)
- Prawdopodobieństwo kryzysu (0-1)
- Wynik kryzysu bazujący na leksykonie
- Wynik niepewności
- Wykryte słowa kluczowe
- Wykryte frazy niepewności

### 6. Wizualizacja wyników

System tworzy kilka wizualizacji przy użyciu funkcji z visualize.py:

```python
create_visualization(all_results, test_texts)
```

#### a) Prawdopodobieństwo kryzysu i sentyment

Wykres słupkowy prawdopodobieństwa kryzysu z nałożoną linią sentymentu dla każdego przykładu.

#### b) Liczba wykrytych słów kluczowych i fraz niepewności

Wykres słupkowy pokazujący liczbę wykrytych terminów związanych z kryzysem i fraz niepewności.

#### c) Analiza wieloczynnikowa

Wykres punktowy przedstawiający zależność między sentymentem a prawdopodobieństwem kryzysu, z kolorem punktów reprezentującym poziom niepewności.

#### d) Wykres radarowy czynników ryzyka

Wykres radarowy porównujący główne czynniki ryzyka (prawdopodobieństwo kryzysu, negatywny sentyment, wynik kryzysu, niepewność).

## Przykład użycia

```python
from dataset import create_sample_dataset
from predictor import EconomicCrisisPredictor
from visualize import create_visualization

# Utwórz zbiór danych i wytrenuj model
texts, labels = create_sample_dataset()
predictor = EconomicCrisisPredictor(use_pretrained=True)
predictor.fit(texts, labels)

# Analizuj nowy tekst
test_text = "Mieszane sygnały z gospodarki wywołują niepewność. Spadek produkcji przemysłowej..."
result = predictor.predict(test_text)
print(f"Prawdopodobieństwo kryzysu: {result['crisis_probability']:.2f}")
print(f"Sentyment: {result['sentiment_label']} ({result['sentiment']['compound']:.2f})")
```

## Zależności

System wymaga następujących pakietów Python:

- numpy
- nltk (z zasobami vader_lexicon, punkt, stopwords)
- scikit-learn
- matplotlib
- sentence-transformers (opcjonalnie dla embeddings BERT)

## Ograniczenia i możliwe ulepszenia

1. System jest trenowany na małym zbiorze przykładów (30 tekstów)
2. System może być dostrojony do konkretnych typów kryzysów przez rozszerzenie słownika i fraz
3. Można poprawić jakość analizy sentymentu przez użycie modelu specjalizowanego w języku polskim
4. Można zastosować bardziej zaawansowane modele NLP, takie jak GPT czy LLama

---

Projekt stanowi demonstrację możliwości przewidywania kryzysu ekonomicznego na podstawie analizy tekstu i może służyć jako punkt wyjścia do bardziej zaawansowanych systemów przewidywania zdarzeń gospodarczych.
