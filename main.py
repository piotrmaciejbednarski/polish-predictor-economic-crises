from dataset import create_sample_dataset
from predictor import EconomicCrisisPredictor
from visualize import create_visualization

def main():
    # Tworzenie przykładowego zbioru danych
    texts, labels = create_sample_dataset()
    
    # Inicjalizacja predyktora kryzysów ekonomicznych
    predictor = EconomicCrisisPredictor(use_pretrained=True)
    
    # Trenowanie predyktora na podstawie przykładowych danych
    predictor.fit(texts, labels)
    
    # Przykładowe teksty do przetestowania predyktora
    test_texts = [
        "Panika na rynku nasila się po upadku kluczowego dewelopera nieruchomości. Inwestorzy obawiają się efektu domina w sektorze budowlanym i finansowym, w obliczu gwałtownego spadku dostępności kredytów.  Dodatkowo, gwałtowny wzrost cen energii i surowców, spowodowany napięciami geopolitycznymi, potęguje obawy o stagflację. Nastroje konsumenckie gwałtownie spadają, a media donoszą o masowych zwolnieniach w sektorze małych i średnich przedsiębiorstw.",
        "Mieszane sygnały z gospodarki wywołują niepewność. Spadek produkcji przemysłowej, szczególnie w sektorze motoryzacyjnym, zbiega się z rosnącą inflacją, napędzaną przez wzrost cen żywności i paliw.  Rynek pracy wykazuje oznaki spowolnienia, a liczba ofert pracy maleje.  Rządowe prognozy wzrostu gospodarczego są rewidowane w dół, a analitycy ostrzegają przed potencjalnym spowolnieniem w drugiej połowie roku.  Dodatkowo, niepewność związana z nowymi regulacjami unijnymi dotyczącymi emisji CO2, wpływa negatywnie na nastroje inwestorów.",
        "Polska gospodarka utrzymuje stabilny wzrost. Rekordowo niskie bezrobocie i rosnące płace napędzają konsumpcję.  Sektor usług, w tym turystyka, odnotowuje dynamiczny rozwój. Inwestycje zagraniczne napływają do Polski, a sektor IT i nowoczesnych technologii kontynuuje ekspansję.  Rządowe programy wsparcia dla przedsiębiorców i rodzin przynoszą pozytywne efekty. Mimo globalnej niepewności, polska gospodarka wykazuje odporność i zdolność adaptacji. Dodatkowo, napływ wykwalifikowanych pracowników z Ukrainy wspiera rynek pracy.",
    ]
    
    # Przechowywanie wyników do wizualizacji
    all_results = []
    
    import time
    start_time = time.time()
    
    print("\nPredyktor kryzysów ekonomicznych\n")
    for i, text in enumerate(test_texts):
        result = predictor.predict(text)
        all_results.append(result)

        print(f"Przykład {i+1}:\n{text}\n")
        print(f"Badany tekst: {result['sentiment_label']} (wartość złożona: {result['sentiment']['compound']:.2f})")
        print(f"Prawdopodobieństwo kryzysu: {result['crisis_probability']:.2f}")
        print(f"Wykryte słowa kluczowe (kryzys): {', '.join(result['crisis_keywords']) if result['crisis_keywords'] else 'Brak'}")
        print(f"Słowa kluczowe (niepewność): {', '.join(result['uncertainty_phrases']) if result['uncertainty_phrases'] else 'Brak'}")
        print("-" * 80)
        
    print("%s sekund" % (time.time() - start_time))
    
    # Tworzenie wizualizacji wyników
    create_visualization(all_results, test_texts)

if __name__ == "__main__":
    main()
