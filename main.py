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
        "Po upadku ważnego dewelopera inwestorzy zaczęli się niepokoić, obawiając się, że to może być początek fali bankructw w budownictwie i finansach. Przyczyną jest nagłe ograniczenie dostępu do kredytów. Dodatkowo, wzrost cen energii i surowców spowodowany napięciami na arenie międzynarodowej zwiększa strach przed kryzysem gospodarczym. Ludzie coraz bardziej martwią się sytuacją, a media informują o wielu zwolnieniach w małych i średnich firmach.",
        
        "Niejasne dane o gospodarce wprowadzają niepewność. Produkcja przemysłowa spada, zwłaszcza w motoryzacji, a jednocześnie rośnie inflacja, głównie przez drożejącą żywność i paliwa. Na rynku pracy widać spowolnienie – liczba ofert pracy maleje. Rząd obniża prognozy wzrostu gospodarczego, a eksperci ostrzegają, że w drugiej połowie roku może dojść do recesji. Dodatkowo, inwestorzy obawiają się nowych unijnych przepisów dotyczących emisji CO2, które mogą mieć negatywny wpływ na gospodarkę.",
        
        "Polska gospodarka rozwija się stabilnie. Bezrobocie jest na rekordowo niskim poziomie, a rosnące pensje zachęcają ludzi do wydawania pieniędzy. Sektor usług, zwłaszcza turystyka, szybko się rozwija. Polska przyciąga zagranicznych inwestorów, a branża IT i nowoczesnych technologii wciąż rośnie. Programy rządowe wspierające firmy i rodziny przynoszą dobre efekty. Mimo trudnej sytuacji na świecie polska gospodarka radzi sobie dobrze i potrafi się dostosować. Dodatkowo, napływ pracowników z Ukrainy pomaga utrzymać rynek pracy w dobrej kondycji."
    ]
    
    # Przechowywanie wyników do wizualizacji
    all_results = []
    
    import time
    start_time = time.time()
    
    print("\nPredyktor kryzysów ekonomicznych, Piotr Bednarski")
    print(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    
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
