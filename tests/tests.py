import requests
import json
import time
import sys

# Konfiguracja
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# Przykładowy artykuł do testu
TEST_PAYLOAD = {
    "text": (
        "Ceny mieszkań w Warszawie znowu rosną. Eksperci alarmują, że młodych ludzi nie stać na kredyt. "
        "Deweloperzy tłumaczą podwyżki rosnącymi kosztami materiałów budowlanych oraz brakiem gruntów pod inwestycje. "
        "Szczególnie trudna sytuacja jest na rynku wynajmu, gdzie stawki osiągnęły historyczne maksimum."
    )
}

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[94m",    # Niebieski
        "SUCCESS": "\033[92m", # Zielony
        "ERROR": "\033[91m",   # Czerwony
        "RESET": "\033[0m"
    }
    print(f"{colors.get(status, '')}[{status}] {message}{colors['RESET']}")

def check_health():
    """Sprawdza czy serwer w ogóle żyje"""
    url = f"{BASE_URL}/health"
    print_status(f"Sprawdzam health check: {url}...", "INFO")
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print_status("Serwer jest ONLINE 🟢", "SUCCESS")
            return True
        else:
            print_status(f"Serwer zwrócił błąd: {response.status_code}", "ERROR")
            return False
    except requests.exceptions.ConnectionError:
        print_status("Nie można połączyć się z serwerem. Czy Docker jest uruchomiony?", "ERROR")
        return False

def test_analyze():
    """Wysyła właściwe zapytanie do analizy"""
    url = f"{BASE_URL}/analyze"
    print_status(f"Wysyłam zapytanie do AI: {url}...", "INFO")
    print_status("⏳ To może potrwać kilka-kilkanaście sekund (AI myśli)...", "INFO")

    start_time = time.time()
    
    try:
        response = requests.post(url, json=TEST_PAYLOAD, headers=HEADERS, timeout=120)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print_status(f"Sukces! Czas odpowiedzi: {elapsed_time:.2f}s", "SUCCESS")
            print("\n" + "="*50)
            print("📦 OTRZYMANY JSON:")
            print("="*50)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print("="*50)
            
            # Weryfikacja czy mamy wyniki
            count = data.get("count", 0)
            if count > 0:
                print_status(f"Znaleziono {count} zdjęć pasujących do artykułu.", "SUCCESS")
            else:
                print_status("Serwis działa, ale nie znalazł pasujących zdjęć (pusta lista).", "INFO")
                
        else:
            print_status(f"Błąd HTTP {response.status_code}", "ERROR")
            print(response.text)

    except requests.exceptions.ReadTimeout:
        print_status("Przekroczono limit czasu (Timeout). Model ładuje się zbyt długo.", "ERROR")
    except Exception as e:
        print_status(f"Wystąpił wyjątek: {e}", "ERROR")

if __name__ == "__main__":
    print("--- ROZPOCZYNAM TESTY SMART IMAGE ROUTER ---\n")
    test_analyze()
