def create_sample_dataset():
    """Utwórz przykładowy zestaw danych do testowania predyktora kryzysu"""
    
    # Format: (text, is_crisis_label) gdzie is_crisis_label: 1 oznacza kryzys, 0 oznacza brak kryzysu
    dataset = [
        # Jasne przykłady kryzysu (label=1)
        ("Gwałtowny wzrost bezrobocia, panika na rynkach finansowych i masowe bankructwa firm sygnalizują głęboki kryzys gospodarczy. Rząd ogłasza stan wyjątkowy.", 1),
        ("Inflacja osiąga rekordowe poziomy, a siła nabywcza obywateli gwałtownie spada. Brakuje podstawowych towarów, a protesty społeczne narastają.", 1),
        ("Upadek kluczowego sektora przemysłu powoduje efekt domina, prowadząc do masowych zwolnień i zamykania fabryk. Kraj pogrąża się w recesji.", 1),
        ("Kryzys zadłużenia publicznego zmusza rząd do drastycznych cięć wydatków i podwyżek podatków. Gospodarka kurczy się, a inwestorzy wycofują kapitał.", 1),
        ("Wojna handlowa z kluczowym partnerem gospodarczym prowadzi do załamania eksportu i importu. Kraj zmaga się z brakiem surowców i rosnącymi cenami.", 1),
        ("Niestabilność polityczna i zamieszki społeczne odstraszają inwestorów zagranicznych. Kurs waluty gwałtownie spada, a inflacja wymyka się spod kontroli.", 1),
        ("Katastrofa naturalna niszczy infrastrukturę i kluczowe sektory gospodarki. Kraj potrzebuje pilnej pomocy międzynarodowej.", 1),
        ("Globalny kryzys finansowy rozprzestrzenia się, dotykając także krajowy sektor bankowy. Kredyty stają się niedostępne, a firmy bankrutują.", 1),
        ("Epidemia powoduje paraliż gospodarki, zamykanie firm i ograniczenia w handlu. Służba zdrowia jest przeciążona, a społeczeństwo ogarnia panika.", 1),
        ("Bańka spekulacyjna na rynku nieruchomości pęka, prowadząc do masowych bankructw deweloperów i spadku cen mieszkań. Banki tracą płynność.", 1),
        ("Sektor energetyczny załamuje się po nagłym odcięciu dostaw gazu ziemnego. Fabryki zamykają produkcję, a koszty ogrzewania domów gwałtownie rosną.", 1),
        ("Masowe strajki paraliżują transport i przemysł. Gospodarka notuje największy spadek od dekady, a rząd traci stabilność.", 1),
        ("Kryzys płynnościowy w bankach prowadzi do wycofywania depozytów. Wprowadzono ograniczenia wypłat, a zaufanie do systemu finansowego załamuje się.", 1),
        ("Załamanie na giełdzie doprowadziło do utraty miliardów złotych oszczędności. Fundusze emerytalne notują rekordowe straty.", 1),
        ("Hiperinflacja wymyka się spod kontroli, osiągając poziom trzycyfrowy. Waluta krajowa traci na wartości z dnia na dzień.", 1),

        # Przykłady umiarkowanych obaw (label=0)
        ("Spowolnienie wzrostu gospodarczego i rosnąca inflacja budzą obawy o stagflację. Rząd rozważa różne scenariusze polityki gospodarczej.", 0),
        ("Napięcia geopolityczne i niepewność na rynkach surowcowych wpływają na nastroje inwestorów. Prognozy gospodarcze są rewidowane w dół.", 0),
        ("Rosnące koszty energii i surowców powodują presję na zyski firm. Niektóre sektory przemysłu rozważają ograniczenie produkcji.", 0),
        ("Rynek pracy wykazuje oznaki spowolnienia, a liczba ofert pracy maleje. Wzrost płac hamuje, a bezrobocie zaczyna rosnąć.", 0),
        ("Niepewność związana z nowymi regulacjami unijnymi wpływa na nastroje przedsiębiorców. Inwestycje w niektóre sektory są wstrzymywane.", 0),
        ("Poziom zadłużenia gospodarstw domowych rośnie, co może ograniczyć przyszłą konsumpcję. Bank centralny monitoruje sytuację.", 0),
        ("Kurs waluty wykazuje oznaki niestabilności, co wpływa na koszty importu i eksportu. Rząd uspokaja rynki, ale niepewność pozostaje.", 0),
        ("Sektor bankowy wykazuje oznaki słabnięcia, a niektóre mniejsze banki mają problemy z płynnością. Regulatorzy finansowi interweniują.", 0),
        ("Zakłócenia w łańcuchach dostaw nadal powodują opóźnienia i wzrost kosztów. Firmy szukają alternatywnych źródeł zaopatrzenia.", 0),
        ("Nastroje konsumenckie spadają, a sprzedaż detaliczna wykazuje oznaki spowolnienia. Klienci ograniczają wydatki na dobra luksusowe.", 0),
        ("Koszt kredytów hipotecznych wzrasta, co wpływa na spowolnienie rynku nieruchomości. Deweloperzy ograniczają nowe inwestycje.", 0),
        ("Rolnicy protestują z powodu rosnących kosztów i niskich cen skupu. Problemy w sektorze rolnym mogą wpłynąć na ceny żywności.", 0),
        ("Wskaźniki wyprzedzające koniunktury wskazują na możliwe spowolnienie w nadchodzących kwartałach. Ekonomiści zalecają ostrożność.", 0),
        ("Deficyt budżetowy przekracza planowany poziom, co budzi obawy o stabilność finansów publicznych. Agencje ratingowe ostrzegają.", 0),
        ("Wymiana handlowa z ważnymi partnerami wykazuje pierwsze oznaki osłabienia. Eksporterzy raportują mniejsze zamówienia.", 0),

        # Przykłady neutralne/pozytywne (label=0)
        ("Gospodarka wykazuje odporność na globalne wstrząsy, a wzrost PKB przekracza oczekiwania. Inwestycje zagraniczne napływają do kraju.", 0),
        ("Inflacja spada, a bank centralny obniża stopy procentowe, stymulując wzrost gospodarczy. Rynek pracy pozostaje silny.", 0),
        ("Sektor technologiczny rozwija się dynamicznie, tworząc nowe miejsca pracy i przyciągając inwestycje. Eksport usług cyfrowych rośnie.", 0),
        ("Rząd wdraża reformy strukturalne, które poprawiają konkurencyjność gospodarki. Inwestorzy reagują pozytywnie na stabilność polityki.", 0),
        ("Wskaźniki zaufania konsumentów i przedsiębiorców osiągają rekordowe poziomy. Inwestycje w infrastrukturę rosną.", 0),
        ("Rynek nieruchomości pozostaje stabilny, a ceny mieszkań rosną umiarkowanie. Banki oferują korzystne warunki kredytowe.", 0),
        ("Eksport rośnie, a bilans handlowy jest dodatni. Kraj staje się ważnym graczem na rynkach międzynarodowych.", 0),
        ("Sektor turystyczny odnotowuje rekordowe wyniki, przyciągając licznych turystów zagranicznych. Inwestycje w infrastrukturę turystyczną rosną.", 0),
        ("Innowacje i rozwój technologiczny napędzają wzrost produktywności. Firmy inwestują w nowoczesne technologie i badania.", 0),
        ("Rząd realizuje programy wsparcia dla małych i średnich przedsiębiorstw, które przyczyniają się do tworzenia nowych miejsc pracy.", 0),
        ("Nowe inwestycje zagraniczne tworzą tysiące miejsc pracy w regionach o wysokim bezrobociu. Rząd oferuje zachęty podatkowe.", 0),
        ("Giełda bije rekordy notowań, a inwestorzy są optymistycznie nastawieni. Kapitalizacja spółek krajowych znacząco rośnie.", 0),
        ("Nowe technologie w energetyce prowadzą do obniżenia kosztów energii. Firmy notują wyższe zyski dzięki niższym kosztom operacyjnym.", 0),
        ("Start-upy przyciągają rekordowe finansowanie venture capital. Polska staje się regionalnym hubem innowacji.", 0),
        ("Dochody budżetu państwa przekraczają prognozy dzięki rosnącemu PKB i skuteczniejszemu poborowi podatków. Deficyt maleje.", 0),
    ]
    
    # Rozdzielenie tekstów i etykiet przed zwróceniem
    sample_texts = [item[0] for item in dataset]
    crisis_labels = [item[1] for item in dataset]
    
    return sample_texts, crisis_labels