import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_visualization(results, texts):
    """
    Tworzy wizualizacje dla wyników przewidywania kryzysu

    Parametry:
    -----------
    results: lista słowników
        Lista wyników przewidywania z modelu
    texts: lista ciągów znaków
        Lista tekstów ekonomicznych, które zostały przeanalizowane
    """
    # Wyciągnij dane do wizualizacji
    probabilities = [r['crisis_probability'] for r in results]
    sentiments = [r['sentiment']['compound'] for r in results]

    # Ustaw figurę z dwoma podwykresami
    fig = plt.figure(figsize=(14, 10))

    # 1. Prawdopodobieństwo kryzysu i sentyment
    ax1 = fig.add_subplot(221)
    x_pos = np.arange(len(texts))
    width = 0.35

    ax1.bar(x_pos, probabilities, width, label='Prawdopodobieństwo kryzysu', color='#FF9999')
    ax1.set_ylabel('Prawdopodobieństwo kryzysu')
    ax1.set_title('Prawdopodobieństwo kryzysu i wynik sentymentu')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Przykład {i+1}' for i in range(len(texts))])
    ax1.set_ylim([0, 1.0])

    # Dodaj sentyment jako linię na osi wtórnej
    ax2 = ax1.twinx()
    ax2.plot(x_pos, sentiments, 'o-', color='blue', label='Wynik sentymentu')
    ax2.set_ylabel('Wynik sentymentu')
    ax2.set_ylim([-1.0, 1.0])

    # Dodaj legendę
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 2. Wizualizacja słów kluczowych i niepewności
    ax3 = fig.add_subplot(222)

    # Policz słowa kluczowe i frazy dla każdego przykładu
    keyword_counts = [len(r['crisis_keywords']) for r in results]
    phrase_counts = [len(r['uncertainty_phrases']) for r in results]

    ax3.bar(x_pos - width/2, keyword_counts, width, label='Kryzys', color='#66B2FF')
    ax3.bar(x_pos + width/2, phrase_counts, width, label='Niepewność', color='#99FF99')
    ax3.set_ylabel('Liczba')
    ax3.set_title('Wykryte słowa kluczowe i frazy niepewności')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Przykład {i+1}' for i in range(len(texts))])
    ax3.legend()

    # 3. Analiza wieloczynnikowa
    ax4 = fig.add_subplot(223)

    # Utwórz wykres punktowy sentymentu vs. prawdopodobieństwo kryzysu
    scatter = ax4.scatter(
        [s['sentiment']['compound'] for s in results],
        [s['crisis_probability'] for s in results],
        c=[s['uncertainty_score'] for s in results],
        s=100,
        cmap='YlOrRd',
        alpha=0.7
    )

    # Dodaj adnotacje tekstowe z numerami przykładów
    for i, txt in enumerate([f"Przykł. {i+1}" for i in range(len(results))]):
        ax4.annotate(txt, (results[i]['sentiment']['compound'], results[i]['crisis_probability']))

    ax4.set_xlabel('Wynik sentymentu')
    ax4.set_ylabel('Prawdopodobieństwo kryzysu')
    ax4.set_title('Prawdopodobieństwo kryzysu vs. sentyment')
    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)  # Próg decyzyjny
    ax4.axvline(x=0, color='r', linestyle='--', alpha=0.3)  # Neutralny sentyment
    ax4.set_xlim([-1.0, 1.0])
    ax4.set_ylim([0, 1.0])

    # Dodaj pasek kolorów dla wyniku niepewności
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Wynik niepewności')

    # 4. Wykres radarowy ryzyka
    ax5 = fig.add_subplot(224, polar=True)

    # Przygotuj dane do wykresu radarowego - tylko przykład 1
    categories = ['Prawdopodobieństwo kryzysu', 'Negatywny sentyment', 'Wynik kryzysu', 'Niepewność']

    # Dla wszystkich przykładów utwórz wykres radarowy
    for i, result in enumerate(results):
        values = [
            result['crisis_probability'],
            max(0, -result['sentiment']['compound']),  # Konwertuj sentyment na wartość dodatnią dla negatywnego sentymentu
            result['crisis_score'],
            result['uncertainty_score']
        ]
        values += values[:1]  # Zamknij pętlę

        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Zamknij pętlę

        color = plt.cm.jet(i/len(results))
        ax5.plot(angles, values, 'o-', linewidth=2, label=f'Przykład {i+1}', color=color)
        ax5.fill(angles, values, alpha=0.1, color=color)

    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories)
    ax5.set_title('Porównanie czynników ryzyka')
    ax5.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.show()
