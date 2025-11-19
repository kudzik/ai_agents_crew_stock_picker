# Kurs CrewAI: Budowa Zaawansowanego Systemu Multi-Agentowego - Stock Picker

## 📚 Wprowadzenie

Witaj w kompleksowym kursie budowy zaawansowanego systemu multi-agentowego wykorzystującego CrewAI. W tym kursie nauczysz się, jak stworzyć profesjonalny system agentów AI, które współpracują ze sobą w celu rozwiązania złożonego problemu: wyboru najlepszej firmy do inwestycji w określonym sektorze.

### Czego się nauczysz?

Po ukończeniu tego kursu będziesz potrafił:

- **Definiować agentów AI** z określonymi rolami, celami i narzędziami
- **Tworzyć zadania** z kontekstem i ustrukturyzowanymi wyjściami
- **Implementować proces hierarchiczny** z managerem LLM
- **Budować niestandardowe narzędzia** dla agentów
- **Konfigurować system pamięci** (krótko- i długoterminową)
- **Wymuszać ustrukturyzowane wyjścia** używając Pydantic

### Wymagania wstępne

- Podstawowa znajomość Pythona (3.10+)
- Podstawowa znajomość YAML
- Zrozumienie koncepcji LLM (Large Language Models)
- Konto OpenAI z kluczem API (lub inny dostawca LLM)

---

## 🏗️ Część 1: Podstawy CrewAI i Struktura Projektu

### 1.1 Czym jest CrewAI?

CrewAI to framework do budowy systemów multi-agentowych, gdzie wiele agentów AI współpracuje ze sobą, aby rozwiązać złożone zadania. Każdy agent ma określoną rolę, cel i dostępne narzędzia, co pozwala na specjalizację i efektywną współpracę.

### 1.2 Kluczowe Koncepcje

#### Agent

Agent to jednostka AI z określoną rolą, celem i backstory. Może mieć dostęp do narzędzi i pamięci.

#### Zadanie (Task)

Zadanie to konkretna praca do wykonania przez agenta. Zadania mogą mieć kontekst (zależności od innych zadań) i ustrukturyzowane wyjścia.

#### Załoga (Crew)

Załoga to zbiór agentów i zadań, które współpracują w określonym procesie (sekwencyjnym lub hierarchicznym).

#### Proces

Proces określa sposób, w jaki zadania są wykonywane:

- **Sekwencyjny**: Zadania wykonywane w ustalonej kolejności
- **Hierarchiczny**: Manager LLM autonomicznie decyduje o kolejności i delegacji

### 1.3 Struktura Projektu Stock Picker

```
ai_agents_crew_stock_picker/
├── src/
│   └── ai_agents_crew_stock_picker/
│       ├── __init__.py
│       ├── crew.py              # Główna definicja załogi
│       ├── main.py              # Punkt wejścia
│       ├── config/
│       │   ├── agents.yaml      # Konfiguracja agentów
│       │   └── tasks.yaml       # Konfiguracja zadań
│       └── tools/
│           ├── __init__.py
│           └── push_tool.py     # Niestandardowe narzędzie
├── docs/
│   └── KURS.MD                  # Ten kurs
├── memory/                      # Automatycznie generowany folder pamięci
├── output/                      # Wyniki pracy agentów
├── pyproject.toml              # Konfiguracja projektu
└── README.md
```

---

## 👥 Część 2: Definiowanie Agentów

### 2.1 Koncepcja Agentów

Agenci w CrewAI to specjalizowane jednostki AI, które wykonują określone zadania. Każdy agent ma:

- **Rolę (Role)**: Kim jest agent (np. "Analityk wiadomości finansowych")
- **Cel (Goal)**: Co agent ma osiągnąć
- **Backstory**: Kontekst i doświadczenie agenta
- **Narzędzia (Tools)**: Dostępne narzędzia do wykonywania zadań
- **Pamięć (Memory)**: Możliwość zapamiętywania poprzednich interakcji

### 2.2 Konfiguracja Agentów w YAML

Plik `config/agents.yaml` definiuje wszystkich agentów w systemie:

```yaml
trending_company_finder:
  role: >
    Analityk wiadomości finansowych, który znajduje firmy będące w trendzie w {sector}
  goal: >
    Czytasz najnowsze wiadomości, a następnie znajdujesz 2-3 firmy, które są w trendzie 
    w wiadomościach do dalszej analizy. Zawsze wybierz nowe firmy. 
    Nie wybieraj tej samej firmy dwa razy.
  backstory: >
    Jesteś ekspertem rynku z umiejętnością wybierać najbardziej interesujące firmy 
    na podstawie najnowszych wiadomości. Zauważasz wiele firm, które są w trendzie 
    w wiadomościach. Odpowiadasz po polsku.
  llm: openai/gpt-4o-mini

financial_researcher:
  role: >
    Starszy Analityk Finansowy
  goal: >
    Na podstawie szczegółów firm w trendzie w wiadomościach, dostarczasz kompleksową 
    analizę każdej firmy w raporcie.
  backstory: >
    Jesteś ekspertem finansowym z doświadczeniem w głębokiej analizie gorących firm 
    i budowaniu kompleksowych raportów. Odpowiadasz po polsku.
  llm: openai/gpt-4o-mini

stock_picker:
  role: >
    Wybieracz akcji z badań
  goal: >
    Na podstawie listy firm z potencjałem inwestycyjnym, wybieraj najlepszą firmę 
    do inwestycji, powiadamiając użytkownika i następnie dostarczając szczegółowy raport. 
    Nie wybieraj tej samej firmy dwa razy.
  backstory: >
    Jesteś precyzyjnym, doświadczonym analitykiem finansowym z umiejętnością wyboru 
    najlepszych akcji. Masz umiejętność syntezy badań i wyboru najlepszej firmy 
    do inwestycji. Odpowiadasz po polsku.
  llm: openai/gpt-4o-mini

manager:
  role: >
    Manager
  goal: >
    Jesteś doświadczonym menedżerem projektu, który może delegować zadania w celu 
    osiągnięcia swojego celu, którym jest wybranie najlepszej firmy do inwestycji.
  backstory: >
    Jesteś doświadczonym i wysoce efektywnym menedżerem projektu, który może 
    delegować zadania do odpowiednich osób. Odpowiadasz po polsku.
  llm: openai/gpt-4o-mini
```

### 2.3 Implementacja Agentów w Pythonie

W pliku `crew.py` agenci są definiowani jako metody z dekoratorem `@agent`:

```python
@agent
def trending_company_finder(self) -> Agent:
    """
    Agent odpowiedzialny za wyszukiwanie firm w trendzie w wiadomościach.
    
    Narzędzia:
        - SerperDevTool: Narzędzie do wyszukiwania w internecie
    
    Pamięć:
        - Włączona: Agent pamięta wcześniej znalezione firmy,
          aby unikać duplikatów w kolejnych uruchomieniach
    """
    return Agent(
        config=self.agents_config["trending_company_finder"],
        tools=[SerperDevTool()],
        memory=True,  # Włączona pamięć, aby unikać duplikatów
    )
```

**Kluczowe elementy:**

1. **Dekorator `@agent`**: Automatycznie rejestruje agenta w systemie CrewAI
2. **`config`**: Ładuje konfigurację z pliku YAML
3. **`tools`**: Lista narzędzi dostępnych dla agenta
4. **`memory`**: Włącza/wyłącza pamięć dla agenta

### 2.4 Wybór Modelu LLM

Każdy agent może używać innego modelu LLM. W projekcie Stock Picker używamy:

- `openai/gpt-4o-mini` dla agentów wykonawczych (szybszy, tańszy)
- Można użyć `openai/gpt-4` dla managera (bardziej zaawansowany, droższy)

---

## 📋 Część 3: Definiowanie Zadań

### 3.1 Koncepcja Zadań

Zadania to konkretne prace do wykonania przez agentów. Każde zadanie ma:

- **Opis (Description)**: Co należy zrobić
- **Oczekiwany wynik (Expected Output)**: Co powinno być zwrócone
- **Agent**: Który agent wykonuje zadanie
- **Kontekst (Context)**: Zależności od innych zadań
- **Plik wyjściowy (Output File)**: Gdzie zapisać wynik

### 3.2 Konfiguracja Zadań w YAML

Plik `config/tasks.yaml` definiuje wszystkie zadania:

```yaml
find_trending_companies:
  description: >
    Znajdź najbardziej popularne firmy w trendzie w wiadomościach w {sector} 
    poprzez przeszukiwanie najnowszych wiadomości. Znajdź nowe firmy, 
    których nie znaleźliście wcześniej.
  expected_output: >
    Lista firm w trendzie w wiadomościach w {sector}
  agent: trending_company_finder
  output_file: output/trending_companies.json

research_trending_companies:
  description: >
    Na podstawie listy firm w trendzie w wiadomościach, dostarcz szczegółową 
    analizę każdej firmy w raporcie poprzez przeszukiwanie internetu
  expected_output: >
    Raport zawierający szczegółową analizę każdej firmy
  agent: financial_researcher
  context:
    - find_trending_companies  # To zadanie otrzymuje wyniki z poprzedniego
  output_file: output/research_report.json

pick_best_company:
  description: >
    Analizuj wyniki badań i wybierz najlepszą firmę do inwestycji.
    Wyślij powiadomienie do użytkownika z decyzją i 1 zdaniem uzasadnienia.
    Następnie odpowiedz szczegółowym raportem po polsku na temat, dlaczego 
    wybraliście tę firmę, a które firmy nie zostały wybrane.
  expected_output: >
    Wybrana firma i dlaczego została wybrana; firmy, które nie zostały wybrane 
    i dlaczego nie zostały wybrane.
  agent: stock_picker
  context:
    - research_trending_companies  # To zadanie otrzymuje analizy firm
  output_file: output/decision.md
```

### 3.3 Kontekst Zadań

**Kontekst** to kluczowa koncepcja w CrewAI. Określa zależności między zadaniami:

- Zadanie `research_trending_companies` ma kontekst `find_trending_companies`
- Oznacza to, że otrzyma wyniki z poprzedniego zadania jako dane wejściowe
- Zapewnia to płynny przepływ danych między zadaniami

### 3.4 Implementacja Zadań w Pythonie

```python
@task
def find_trending_companies(self) -> Task:
    """
    Zadanie: Znajdź firmy w trendzie w wiadomościach.
    
    Wykonawca: trending_company_finder
    Wyjście: Ustrukturyzowane (TrendingCompanyList)
    """
    return Task(
        config=self.tasks_config["find_trending_companies"],
        output_pydantic=TrendingCompanyList,  # Wymusza ustrukturyzowane wyjście
    )
```

---

## 🔧 Część 4: Ustrukturyzowane Wyjścia z Pydantic

### 4.1 Dlaczego Ustrukturyzowane Wyjścia?

LLM mogą zwracać dane w nieprzewidywalnym formacie tekstowym. Ustrukturyzowane wyjścia wymuszają zwracanie danych w określonym formacie JSON, co zapewnia:

- **Stabilność**: Zawsze otrzymujesz dane w tym samym formacie
- **Łatwość przetwarzania**: Dane są automatycznie przekształcane w obiekty Pythona
- **Walidację**: Pydantic automatycznie waliduje dane

### 4.2 Definiowanie Schematów Pydantic

W pliku `crew.py` definiujemy schematy Pydantic:

```python
from pydantic import BaseModel, Field
from typing import List

class TrendingCompany(BaseModel):
    """
    Reprezentuje pojedynczą firmę, która jest w trendzie w wiadomościach.
    """
    name: str = Field(description="Nazwa firmy")
    ticker: str = Field(description="Symbol giełdowy firmy (np. AAPL, MSFT)")
    reason: str = Field(description="Powód, dlaczego firma jest w wiadomościach")

class TrendingCompanyList(BaseModel):
    """
    Lista firm w trendzie w wiadomościach.
    """
    companies: List[TrendingCompany] = Field(
        description="Lista firm, o których jest głośno w wiadomościach"
    )
```

**Kluczowe elementy:**

1. **`BaseModel`**: Bazowa klasa Pydantic dla wszystkich schematów
2. **`Field`**: Definiuje pole z opisem dla LLM
3. **`description`**: Opis pola pomaga LLM zrozumieć, co należy wypełnić

### 4.3 Użycie w Zadaniach

```python
@task
def find_trending_companies(self) -> Task:
    return Task(
        config=self.tasks_config["find_trending_companies"],
        output_pydantic=TrendingCompanyList,  # Wymusza format JSON
    )
```

**Efekt:** Agent zwróci dane w formacie JSON zgodnym ze schematem, które są automatycznie przekształcane w obiekt Pythona typu `TrendingCompanyList`.

### 4.4 Przykład Wyniku

Po wykonaniu zadania, plik `output/trending_companies.json` będzie zawierał:

```json
{
  "companies": [
    {
      "name": "Apple Inc.",
      "ticker": "AAPL",
      "reason": "Nowy produkt iPhone 15 generuje duże zainteresowanie"
    },
    {
      "name": "Microsoft Corporation",
      "ticker": "MSFT",
      "reason": "Wzrost w chmurze Azure i AI"
    }
  ]
}
```

---

## 🛠️ Część 5: Niestandardowe Narzędzia

### 5.1 Czym są Narzędzia?

Narzędzia to funkcje, które agenci mogą wywoływać podczas wykonywania zadań. CrewAI oferuje wiele wbudowanych narzędzi (np. `SerperDevTool` do wyszukiwania), ale możesz również tworzyć własne.

### 5.2 Tworzenie Niestandardowego Narzędzia

W projekcie Stock Picker stworzyliśmy narzędzie do wysyłania powiadomień push. Oto jak to działa:

#### Krok 1: Definiowanie Schematu Wejściowego

```python
from pydantic import BaseModel, Field

class PushNotification(BaseModel):
    """
    Schemat danych wejściowych dla narzędzia powiadomień push.
    """
    message: str = Field(
        ...,
        description="Treść powiadomienia do wysłania do użytkownika."
    )
```

#### Krok 2: Tworzenie Klasy Narzędzia

```python
from crewai.tools import BaseTool
from typing import Type

class PushNotificationTool(BaseTool):
    """
    Niestandardowe narzędzie do wysyłania powiadomień push.
    """
    name: str = "Wyślij powiadomienie"
    description: str = (
        "Narzędzie do wysyłania powiadomień push do użytkownika. "
        "Użyj tego narzędzia, gdy chcesz powiadomić użytkownika o ważnej decyzji."
    )
    args_schema: Type[BaseModel] = PushNotification

    def _run(self, message: str) -> str:
        """
        Wykonuje wysłanie powiadomienia push.
        """
        # Implementacja logiki wysyłania powiadomienia
        # ...
        return '{"notification": "ok"}'
```

**Kluczowe elementy:**

1. **`BaseTool`**: Bazowa klasa dla wszystkich narzędzi CrewAI
2. **`name`**: Nazwa narzędzia widoczna dla LLM
3. **`description`**: Opis używany przez LLM do decyzji o użyciu
4. **`args_schema`**: Schemat Pydantic definiujący parametry
5. **`_run()`**: Metoda zawierająca faktyczną logikę narzędzia

#### Krok 3: Pełna Implementacja

```python
import os
import requests

def _run(self, message: str) -> str:
    # Pobierz dane uwierzytelniające z zmiennych środowiskowych
    pushover_user = os.getenv("PUSHOVER_USER")
    pushover_token = os.getenv("PUSHOVER_TOKEN")
    pushover_url = os.getenv(
        "PUSHOVER_URL", 
        "https://api.pushover.net/1/messages.json"
    )

    # Walidacja
    if not pushover_user or not pushover_token:
        return '{"error": "Brak wymaganych zmiennych środowiskowych"}'

    # Przygotuj dane
    payload = {
        "user": pushover_user,
        "token": pushover_token,
        "message": message,
    }

    # Wyślij żądanie
    response = requests.post(pushover_url, data=payload, timeout=10)
    response.raise_for_status()
    
    return '{"notification": "ok"}'
```

### 5.3 Przypisanie Narzędzia do Agenta

```python
@agent
def stock_picker(self) -> Agent:
    return Agent(
        config=self.agents_config["stock_picker"],
        tools=[PushNotificationTool()],  # Dodaj narzędzie
        memory=True,
    )
```

**Jak to działa:**

1. Agent otrzymuje opis narzędzia w swoim kontekście
2. Podczas wykonywania zadania, agent **autonomicznie decyduje**, czy użyć narzędzia
3. Jeśli zdecyduje się użyć, wywołuje metodę `_run()` z odpowiednimi parametrami
4. Wynik jest zwracany do agenta jako część kontekstu

---

## 🏛️ Część 6: Proces Hierarchiczny

### 6.1 Proces Sekwencyjny vs Hierarchiczny

#### Proces Sekwencyjny

- Zadania wykonywane w **ustalonej kolejności**
- Prosty i przewidywalny
- Brak elastyczności

#### Proces Hierarchiczny

- **Manager LLM** autonomicznie decyduje o kolejności zadań
- Manager **deleguje** zadania do odpowiednich agentów
- Elastyczny i adaptacyjny
- Bardziej zaawansowany, ale mniej przewidywalny

### 6.2 Definiowanie Managera

Manager to specjalny agent, który koordynuje pracę załogi:

```python
@crew
def crew(self) -> Crew:
    # Tworzenie managera
    manager = Agent(
        config=self.agents_config["manager"],
        allow_delegation=True,  # Manager może delegować zadania
    )
    
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.hierarchical,  # Proces hierarchiczny
        manager_agent=manager,  # Przypisanie managera
        # ...
    )
```

**Kluczowe elementy:**

1. **`allow_delegation=True`**: Umożliwia managerowi delegowanie zadań
2. **`Process.hierarchical`**: Ustawia proces hierarchiczny
3. **`manager_agent`**: Przypisuje managera do załogi

### 6.3 Jak Działa Proces Hierarchiczny?

1. **Inicjalizacja**: Manager otrzymuje listę wszystkich zadań i agentów
2. **Analiza**: Manager analizuje cele załogi i dostępne zadania
3. **Decyzja**: Manager **autonomicznie decyduje**, które zadanie wykonać i który agent jest najlepszy
4. **Delegacja**: Manager deleguje zadanie do wybranego agenta
5. **Wykonanie**: Agent wykonuje zadanie i zwraca wynik
6. **Iteracja**: Proces powtarza się, aż wszystkie zadania zostaną wykonane

**Przykład przepływu:**

```
Manager: "Muszę znaleźć firmy w trendzie. Deleguję zadanie 'find_trending_companies' 
         do agenta 'trending_company_finder'."

Trending Company Finder: [Wykonuje zadanie, zwraca listę firm]

Manager: "Mam listę firm. Teraz potrzebuję analizy. Deleguję zadanie 
         'research_trending_companies' do agenta 'financial_researcher'."

Financial Researcher: [Wykonuje zadanie, zwraca analizy]

Manager: "Mam analizy. Teraz potrzebuję wyboru najlepszej firmy. Deleguję zadanie 
         'pick_best_company' do agenta 'stock_picker'."

Stock Picker: [Wykonuje zadanie, wybiera firmę, wysyła powiadomienie]
```

### 6.4 Zalety Procesu Hierarchicznego

- **Elastyczność**: Manager może zmieniać kolejność zadań w zależności od sytuacji
- **Optymalizacja**: Manager wybiera najlepszego agenta dla każdego zadania
- **Adaptacyjność**: System może reagować na nieoczekiwane sytuacje
- **Skalowalność**: Łatwo dodać nowe zadania i agentów

### 6.5 Wady Procesu Hierarchicznego

- **Nieprzewidywalność**: Kolejność zadań może być różna przy każdym uruchomieniu
- **Złożoność**: Wymaga bardziej zaawansowanego managera (np. GPT-4)
- **Koszty**: Manager wykonuje dodatkowe wywołania LLM

---

## 💾 Część 7: System Pamięci

### 7.1 Dlaczego Pamięć?

Agenci AI działają w kontekście pojedynczej interakcji. Pamięć pozwala agentom:

- **Zapamiętywać** poprzednie interakcje
- **Unikać duplikatów** (np. nie wybierać tej samej firmy dwa razy)
- **Budować wiedzę** na podstawie wcześniejszych doświadczeń
- **Uczyć się** z historii

### 7.2 Typy Pamięci w CrewAI

CrewAI oferuje trzy główne typy pamięci:

| Typ Pamięci | Cel | Implementacja |
|------------|-----|---------------|
| **Krótkoterminowa (Short-term)** | Przechowywanie ostatnich interakcji | RAG (ChromaDB) |
| **Długoterminowa (Long-term)** | Przechowywanie ważnych informacji | SQLite |
| **Encji (Entity)** | Przechowywanie informacji o konkretnych rzeczach | RAG (ChromaDB) |

### 7.3 Konfiguracja Pamięci

```python
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage

@crew
def crew(self) -> Crew:
    # Pamięć długoterminowa (SQLite)
    long_term_memory = LongTermMemory(
        storage=LTMSQLiteStorage(
            db_path="./memory/long_term_mem_store.db"
        )
    )

    # Pamięć krótkoterminowa (RAG/ChromaDB)
    short_term_memory = ShortTermMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "openai",
                "model": "text-embedding-3-small",  # Model do embeddings
            },
            type="short_term",
            path="./memory",
        )
    )

    # Pamięć encji (RAG/ChromaDB)
    entity_memory = EntityMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "openai",
                "model": "text-embedding-3-small",
            },
            type="short_term",
            path="./memory",
        )
    )

    return Crew(
        # ...
        memory=True,  # Włącza system pamięci
        long_term_memory=long_term_memory,
        short_term_memory=short_term_memory,
        entity_memory=entity_memory,
    )
```

### 7.4 Włączanie Pamięci dla Agentów

Nie wystarczy włączyć pamięci w załodze - każdy agent musi mieć ją włączoną osobno:

```python
@agent
def trending_company_finder(self) -> Agent:
    return Agent(
        config=self.agents_config["trending_company_finder"],
        tools=[SerperDevTool()],
        memory=True,  # Włączona pamięć dla tego agenta
    )
```

### 7.5 Jak Działa Pamięć?

1. **Zapis**: Gdy agent wykonuje zadanie, informacje są automatycznie zapisywane w pamięci
2. **Wyszukiwanie**: Przed wykonaniem zadania, agent wyszukuje podobne konteksty w pamięci
3. **Wstrzykiwanie**: Znalezione konteksty są wstrzykiwane do promptu systemowego agenta
4. **Użycie**: Agent używa tych informacji do podejmowania decyzji

**Przykład:**

```
Agent Stock Picker przed wyborem firmy:
1. Wyszukuje w pamięci: "Jakie firmy wybrałem wcześniej?"
2. Znajduje: "Wcześniej wybrałem Apple Inc. (AAPL)"
3. Otrzymuje w kontekście: "Nie wybieraj Apple Inc. ponownie"
4. Wybiera inną firmę (np. Microsoft)
```

### 7.6 RAG (Retrieval-Augmented Generation)

Pamięć krótkoterminowa i encji używają techniki RAG:

1. **Embeddings**: Tekst jest przekształcany w wektory numeryczne (embeddings)
2. **Przechowywanie**: Wektory są przechowywane w bazie wektorowej (ChromaDB)
3. **Wyszukiwanie**: Podczas wyszukiwania, zapytanie jest również przekształcane w embedding
4. **Podobieństwo**: System znajduje najbardziej podobne konteksty (używając podobieństwa cosinusowego)
5. **Zwrot**: Znalezione konteksty są zwracane do agenta

**Wymagania:**

- Model embeddings (np. `text-embedding-3-small` z OpenAI)
- Dostęp do API embeddings (wymaga klucza API)

---

## 🚀 Część 8: Uruchomienie i Użycie

### 8.1 Konfiguracja Środowiska

#### Krok 1: Instalacja Zależności

```bash
# Zainstaluj UV (jeśli nie masz)
pip install uv

# Zainstaluj zależności projektu
crewai install
```

#### Krok 2: Konfiguracja Zmiennych Środowiskowych

Utwórz plik `.env` w katalogu głównym projektu:

```env
# Wymagane
OPENAI_API_KEY=sk-...

# Opcjonalne (dla narzędzia Pushover)
PUSHOVER_USER=twoj_user_id
PUSHOVER_TOKEN=twoj_token
PUSHOVER_URL=https://api.pushover.net/1/messages.json

# Opcjonalne (dla SerperDevTool)
SERPER_API_KEY=twoj_serper_key
```

#### Krok 3: Konfiguracja SerperDevTool

`SerperDevTool` wymaga klucza API z [Serper.dev](https://serper.dev). Dodaj klucz do `.env`.

### 8.2 Uruchomienie Projektu

#### Metoda 1: Użycie CrewAI CLI

```bash
crewai run
```

#### Metoda 2: Bezpośrednie Uruchomienie Pythona

```bash
python -m ai_agents_crew_stock_picker.main
```

#### Metoda 3: Modyfikacja Danych Wejściowych

Edytuj plik `main.py`:

```python
inputs = {
    "sector": "technology",  # Zmień na inny sektor
    "region": "Europe",      # Zmień region
}
```

### 8.3 Struktura Wyników

Po uruchomieniu, w folderze `output/` znajdziesz:

- **`trending_companies.json`**: Lista firm w trendzie (ustrukturyzowane wyjście)
- **`research_report.json`**: Szczegółowe analizy firm (ustrukturyzowane wyjście)
- **`decision.md`**: Raport decyzyjny z wyborem najlepszej firmy

### 8.4 Interpretacja Wyników

#### Plik `trending_companies.json`

```json
{
  "companies": [
    {
      "name": "Apple Inc.",
      "ticker": "AAPL",
      "reason": "Nowy produkt iPhone 15 generuje duże zainteresowanie"
    }
  ]
}
```

#### Plik `research_report.json`

```json
{
  "research_list": [
    {
      "name": "Apple Inc.",
      "market_position": "Lider w sektorze technologicznym...",
      "future_outlook": "Pozytywne perspektywy...",
      "investment_potential": "Wysoki potencjał inwestycyjny..."
    }
  ]
}
```

#### Plik `decision.md`

Markdown raport zawierający:

- Wybraną firmę i uzasadnienie
- Firmy, które nie zostały wybrane i dlaczego
- Szczegółową analizę decyzji

---

## 🎯 Część 9: Najlepsze Praktyki

### 9.1 Projektowanie Agentów

✅ **DO:**

- Definiuj jasne role i cele
- Używaj szczegółowych backstory
- Przypisuj odpowiednie narzędzia
- Włączaj pamięć tam, gdzie to potrzebne

❌ **NIE:**

- Nie twórz zbyt ogólnych agentów
- Nie przypisuj zbyt wielu narzędzi jednemu agentowi
- Nie włączaj pamięci wszędzie (tylko tam, gdzie jest potrzebna)

### 9.2 Projektowanie Zadań

✅ **DO:**

- Definiuj jasne opisy zadań
- Używaj kontekstu do przekazywania danych
- Wymuszaj ustrukturyzowane wyjścia dla danych krytycznych
- Określaj pliki wyjściowe

❌ **NIE:**

- Nie twórz zbyt ogólnych zadań
- Nie zapominaj o kontekście (zależnościach)
- Nie używaj ustrukturyzowanych wyjść dla wszystkich zadań (tylko tam, gdzie potrzebne)

### 9.3 Ustrukturyzowane Wyjścia

✅ **DO:**

- Używaj szczegółowych opisów w Field()
- Definiuj spójne nazwy pól
- Waliduj dane za pomocą Pydantic

❌ **NIE:**

- Nie używaj zbyt ogólnych opisów
- Nie pomijaj walidacji

### 9.4 Niestandardowe Narzędzia

✅ **DO:**

- Twórz jasne opisy narzędzi
- Waliduj dane wejściowe
- Obsługuj błędy gracefully
- Zwracaj użyteczne komunikaty błędów

❌ **NIE:**

- Nie pomijaj walidacji
- Nie rzucaj nieobsłużonych wyjątków

### 9.5 Proces Hierarchiczny

✅ **DO:**

- Używaj mocniejszego modelu dla managera (np. GPT-4)
- Definiuj jasne cele dla managera
- Testuj różne scenariusze

❌ **NIE:**

- Nie używaj zbyt słabego modelu dla managera
- Nie pomijaj `allow_delegation=True`

### 9.6 Pamięć

✅ **DO:**

- Włączaj pamięć selektywnie (tylko tam, gdzie potrzebna)
- Używaj odpowiedniego modelu embeddings
- Regularnie czyść starą pamięć (jeśli potrzebne)

❌ **NIE:**

- Nie włączaj pamięci wszędzie
- Nie używaj zbyt słabego modelu embeddings

---

## 🔍 Część 10: Debugowanie i Rozwiązywanie Problemów

### 10.1 Typowe Problemy

#### Problem: Agent nie używa narzędzia

**Rozwiązanie:**

- Sprawdź, czy opis narzędzia jest jasny
- Upewnij się, że narzędzie jest przypisane do agenta
- Sprawdź logi (verbose=True) aby zobaczyć, co agent myśli

#### Problem: Ustrukturyzowane wyjście nie działa

**Rozwiązanie:**

- Sprawdź, czy schemat Pydantic jest poprawny
- Upewnij się, że opisy Field() są szczegółowe
- Sprawdź logi, aby zobaczyć, co agent zwraca

#### Problem: Pamięć nie działa

**Rozwiązanie:**

- Sprawdź, czy pamięć jest włączona w załodze
- Sprawdź, czy pamięć jest włączona dla agenta
- Upewnij się, że masz dostęp do modelu embeddings
- Sprawdź, czy folder `memory/` jest tworzony

#### Problem: Manager nie deleguje zadań

**Rozwiązanie:**

- Sprawdź, czy `allow_delegation=True`
- Sprawdź, czy `process=Process.hierarchical`
- Sprawdź, czy manager ma przypisanego agenta
- Sprawdź logi, aby zobaczyć decyzje managera

### 10.2 Włączanie Szczegółowych Logów

```python
return Crew(
    # ...
    verbose=True,  # Włącza szczegółowe logi
)
```

### 10.3 Sprawdzanie Struktury Danych

Dodaj print statements w kodzie:

```python
@task
def find_trending_companies(self) -> Task:
    task = Task(
        config=self.tasks_config["find_trending_companies"],
        output_pydantic=TrendingCompanyList,
    )
    print(f"Task config: {task.config}")  # Debug
    return task
```

---

## 📚 Część 11: Rozszerzanie Projektu

### 11.1 Dodawanie Nowych Agentów

1. Dodaj konfigurację do `agents.yaml`
2. Dodaj metodę `@agent` w `crew.py`
3. Agent będzie automatycznie dostępny w załodze

### 11.2 Dodawanie Nowych Zadań

1. Dodaj konfigurację do `tasks.yaml`
2. Dodaj metodę `@task` w `crew.py`
3. Zadanie będzie automatycznie dostępne w załodze

### 11.3 Dodawanie Nowych Narzędzi

1. Utwórz nowy plik w folderze `tools/`
2. Zdefiniuj klasę dziedziczącą z `BaseTool`
3. Zaimportuj i przypisz do agenta w `crew.py`

### 11.4 Modyfikacja Procesu

Możesz zmienić proces z hierarchicznego na sekwencyjny:

```python
return Crew(
    # ...
    process=Process.sequential,  # Zamiast Process.hierarchical
    # Usuń manager_agent
)
```

---

## 🎓 Podsumowanie

### Kluczowe Koncepcje

1. **Agenci**: Specjalizowane jednostki AI z rolami, celami i narzędziami
2. **Zadania**: Konkretne prace z kontekstem i ustrukturyzowanymi wyjściami
3. **Ustrukturyzowane Wyjścia**: Pydantic schematy wymuszające format JSON
4. **Niestandardowe Narzędzia**: Własne funkcje dla agentów
5. **Proces Hierarchiczny**: Manager LLM koordynujący pracę załogi
6. **Pamięć**: System przechowywania i wyszukiwania kontekstu

### Następne Kroki

1. **Eksperymentuj**: Zmieniaj konfiguracje agentów i zadań
2. **Rozszerzaj**: Dodawaj nowe funkcjonalności
3. **Optymalizuj**: Ulepszaj opisy i konfiguracje
4. **Ucz się**: Czytaj dokumentację CrewAI i eksperymentuj

### Przydatne Linki

- [Dokumentacja CrewAI](https://docs.crewai.com)
- [GitHub CrewAI](https://github.com/joaomdmoura/crewai)
- [Discord CrewAI](https://discord.com/invite/X4JWnZnxPb)

---

## 📝 Ćwiczenia

### Ćwiczenie 1: Dodaj Nowego Agenta

Dodaj agenta "Risk Analyst", który analizuje ryzyko inwestycji w wybranej firmie.

### Ćwiczenie 2: Stwórz Nowe Narzędzie

Stwórz narzędzie do wysyłania emaili z rekomendacją inwestycyjną.

### Ćwiczenie 3: Zmień Proces

Zmień proces z hierarchicznego na sekwencyjny i porównaj wyniki.

### Ćwiczenie 4: Rozszerz Schematy Pydantic

Dodaj nowe pola do schematu `TrendingCompanyResearch` (np. `risk_level`, `price_target`).

---

**Powodzenia w budowaniu własnych systemów multi-agentowych! 🚀**
