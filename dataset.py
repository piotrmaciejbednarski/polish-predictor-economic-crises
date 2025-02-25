def create_sample_dataset():
    """Utwórz przykładowy zestaw danych do testowania predyktora kryzysu"""
    
    sample_texts = [
        # Jasne przykłady kryzysu
        "Gwałtowny wzrost bezrobocia, panika na rynkach finansowych i masowe bankructwa firm sygnalizują głęboki kryzys gospodarczy. Rząd ogłasza stan wyjątkowy.",
        "Inflacja osiąga rekordowe poziomy, a siła nabywcza obywateli gwałtownie spada. Brakuje podstawowych towarów, a protesty społeczne narastają.",
        "Upadek kluczowego sektora przemysłu powoduje efekt domina, prowadząc do masowych zwolnień i zamykania fabryk. Kraj pogrąża się w recesji.",
        "Kryzys zadłużenia publicznego zmusza rząd do drastycznych cięć wydatków i podwyżek podatków. Gospodarka kurczy się, a inwestorzy wycofują kapitał.",
        "Wojna handlowa z kluczowym partnerem gospodarczym prowadzi do załamania eksportu i importu. Kraj zmaga się z brakiem surowców i rosnącymi cenami.",
        "Niestabilność polityczna i zamieszki społeczne odstraszają inwestorów zagranicznych. Kurs waluty gwałtownie spada, a inflacja wymyka się spod kontroli.",
        "Katastrofa naturalna niszczy infrastrukturę i kluczowe sektory gospodarki. Kraj potrzebuje pilnej pomocy międzynarodowej.",
        "Globalny kryzys finansowy rozprzestrzenia się, dotykając także krajowy sektor bankowy. Kredyty stają się niedostępne, a firmy bankrutują.",
        "Epidemia powoduje paraliż gospodarki, zamykanie firm i ograniczenia w handlu. Służba zdrowia jest przeciążona, a społeczeństwo ogarnia panika.",
        "Bańka spekulacyjna na rynku nieruchomości pęka, prowadząc do masowych bankructw deweloperów i spadku cen mieszkań. Banki tracą płynność.",

        # Przykłady umiarkowanych obaw
        "Spowolnienie wzrostu gospodarczego i rosnąca inflacja budzą obawy o stagflację. Rząd rozważa różne scenariusze polityki gospodarczej.",
        "Napięcia geopolityczne i niepewność na rynkach surowcowych wpływają na nastroje inwestorów. Prognozy gospodarcze są rewidowane w dół.",
        "Rosnące koszty energii i surowców powodują presję na zyski firm. Niektóre sektory przemysłu rozważają ograniczenie produkcji.",
        "Rynek pracy wykazuje oznaki spowolnienia, a liczba ofert pracy maleje. Wzrost płac hamuje, a bezrobocie zaczyna rosnąć.",
        "Niepewność związana z nowymi regulacjami unijnymi wpływa na nastroje przedsiębiorców. Inwestycje w niektóre sektory są wstrzymywane.",
        "Poziom zadłużenia gospodarstw domowych rośnie, co może ograniczyć przyszłą konsumpcję. Bank centralny monitoruje sytuację.",
        "Kurs waluty wykazuje oznaki niestabilności, co wpływa na koszty importu i eksportu. Rząd uspokaja rynki, ale niepewność pozostaje.",
        "Sektor bankowy wykazuje oznaki słabnięcia, a niektóre mniejsze banki mają problemy z płynnością. Regulatorzy finansowi interweniują.",
        "Zakłócenia w łańcuchach dostaw nadal powodują opóźnienia i wzrost kosztów. Firmy szukają alternatywnych źródeł zaopatrzenia.",
        "Nastroje konsumenckie spadają, a sprzedaż detaliczna wykazuje oznaki spowolnienia. Klienci ograniczają wydatki na dobra luksusowe.",

        # Przykłady neutralne/pozytywne
        "Gospodarka wykazuje odporność na globalne wstrząsy, a wzrost PKB przekracza oczekiwania. Inwestycje zagraniczne napływają do kraju.",
        "Inflacja spada, a bank centralny obniża stopy procentowe, stymulując wzrost gospodarczy. Rynek pracy pozostaje silny.",
        "Sektor technologiczny rozwija się dynamicznie, tworząc nowe miejsca pracy i przyciągając inwestycje. Eksport usług cyfrowych rośnie.",
        "Rząd wdraża reformy strukturalne, które poprawiają konkurencyjność gospodarki. Inwestorzy reagują pozytywnie na stabilność polityki.",
        "Wskaźniki zaufania konsumentów i przedsiębiorców osiągają rekordowe poziomy. Inwestycje w infrastrukturę rosną.",
        "Rynek nieruchomości pozostaje stabilny, a ceny mieszkań rosną umiarkowanie. Banki oferują korzystne warunki kredytowe.",
        "Eksport rośnie, a bilans handlowy jest dodatni. Kraj staje się ważnym graczem na rynkach międzynarodowych.",
        "Sektor turystyczny odnotowuje rekordowe wyniki, przyciągając licznych turystów zagranicznych. Inwestycje w infrastrukturę turystyczną rosną.",
        "Innowacje i rozwój technologiczny napędzają wzrost produktywności. Firmy inwestują w nowoczesne technologie i badania.",
        "Rząd realizuje programy wsparcia dla małych i średnich przedsiębiorstw, które przyczyniają się do tworzenia nowych miejsc pracy."
    ]

    # Labels: 1 for clear crisis, 0 for non-crisis
    crisis_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    return sample_texts, crisis_labels