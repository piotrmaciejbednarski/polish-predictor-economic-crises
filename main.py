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
        "Po bankructwie kluczowego dewelopera, inwestorzy wpadli w panikę, obawiając się, że to początek serii upadłości w sektorach budowlanym i finansowym, spowodowanych gwałtownym ograniczeniem dostępu do kredytów. Ponadto, rosnące ceny energii i surowców, wynikające z międzynarodowych napięć politycznych, potęgują strach przed stagflacją. Nastroje konsumentów dramatycznie się pogarszają, a media donoszą o licznych zwolnieniach w małych i średnich przedsiębiorstwach.",
        "Niejednoznaczne dane gospodarcze wywołują niepewność. Spadek produkcji przemysłowej, szczególnie w branży motoryzacyjnej, występuje równocześnie z nasilającą się inflacją, która jest napędzana wzrostem cen żywności i paliw. Rynek pracy sygnalizuje spowolnienie, a liczba ofert zatrudnienia maleje. Rządowe prognozy wzrostu gospodarczego są korygowane w dół, a eksperci przewidują możliwość recesji w drugiej połowie roku. Dodatkowo, niepewność związana z nowymi regulacjami Unii Europejskiej dotyczącymi emisji CO2, negatywnie wpływa na nastroje inwestorów.",
        "Polska gospodarka utrzymuje stabilny rozwój. Rekordowo niskie bezrobocie i rosnące wynagrodzenia stymulują wydatki konsumenckie. Sektor usług, w tym turystyka, dynamicznie się rozwija. Do Polski napływają inwestycje z zagranicy, a sektor IT i nowoczesnych technologii kontynuuje swój wzrost. Rządowe programy wsparcia dla przedsiębiorców i rodzin przynoszą pozytywne rezultaty. Pomimo globalnej niepewności, polska gospodarka wykazuje odporność i zdolność do adaptacji. Dodatkowo, napływ wykwalifikowanych pracowników z Ukrainy wspiera rynek pracy."
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
