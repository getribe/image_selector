import os
import json
import asyncio
import aiohttp
import requests
import torch
import logging
import sys
import uvicorn
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List
from PIL import Image
from google.api_core.exceptions import ResourceExhausted

# Logging & Observability
from pythonjsonlogger import jsonlogger

# Starlette imports
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.requests import Request

# ML & AI imports
from transformers import AutoProcessor, AutoModel
import google.generativeai as genai
from dotenv import load_dotenv


# --- KONFIGURACJA LOGOWANIA (Loki & Tempo Friendly) ---

class TraceContextFilter(logging.Filter):
    """
    Filtr dodający trace_id i span_id do każdego rekordu logu.
    Dzięki temu Loki będzie mógł skorelować logi z Tempo.
    """

    def filter(self, record):
        # Domyślne wartości
        record.trace_id = ""
        record.span_id = ""

        try:
            # Próbujemy pobrać kontekst z OpenTelemetry (nawet jak jest wstrzyknięte auto-instrumentation)
            from opentelemetry import trace
            span = trace.get_current_span()
            if span != trace.NonRecordingSpan(None):
                ctx = span.get_span_context()
                if ctx.is_valid:
                    record.trace_id = f"{ctx.trace_id:032x}"
                    record.span_id = f"{ctx.span_id:016x}"
        except ImportError:
            pass  # OTel nie jest zainstalowany lub dostępny
        except Exception:
            pass  # Inny błąd, nie chcemy przerywać logowania

        return True


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    # Format JSON zawierający timestamp, poziom, wiadomość i klucze tracingowe
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s %(trace_id)s %(span_id)s',
        rename_fields={"levelname": "level", "asctime": "timestamp"},
        datefmt='%Y-%m-%dT%H:%M:%SZ'
    )

    handler.setFormatter(formatter)
    handler.addFilter(TraceContextFilter())

    # Usuwamy domyślne handlery (żeby nie dublować logów)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    # Wyciszenie bibliotek, które mogą za bardzo hałasować
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # Uvicorn ma własne logi access
    logging.getLogger("urllib3").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger("ImageService")

# --- KONFIGURACJA ZMIENNYCH ---
load_dotenv()
GOOGLE_GENAI_KEY = os.getenv("GOOGLE_GENAI_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

if GOOGLE_GENAI_KEY:
    genai.configure(api_key=GOOGLE_GENAI_KEY)
else:
    logger.critical("Brak klucza GOOGLE_GENAI_KEY! Aplikacja nie zadziała poprawnie.")


# ==========================================
# CZĘŚĆ 1: TWOJA LOGIKA (KLASY POMOCNICZE)
# ==========================================

class CostCalculator:
    def __init__(self):
        self.total_input = 0
        self.total_output = 0

    def add_gemini_usage(self, usage):
        if hasattr(usage, 'prompt_token_count'): self.total_input += usage.prompt_token_count
        if hasattr(usage, 'candidates_token_count'): self.total_output += usage.candidates_token_count

    def log_summary(self):
        """Loguje zużycie jako JSON, łatwy do parsowania przez Loki"""
        logger.info("Gemini Token Usage Summary", extra={
            "metric_type": "cost",
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output
        })


class LocalVisualRanker:
    def __init__(self):
        logger.info("Inicjalizacja modelu SigLIP...", extra={"event": "model_loading_start"})
        try:
            model_id = "google/siglip-base-patch16-256-multilingual"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"SigLIP gotowy", extra={"event": "model_loading_success", "device": self.device})
        except Exception as e:
            logger.error("Błąd ładowania SigLIP", exc_info=True, extra={"event": "model_loading_error"})
            raise e

    def rank_images(self, text_query: str, images: List[Image.Image]) -> List[float]:
        if not images: return []
        texts = [text_query] * len(images)
        try:
            inputs = self.processor(text=texts, images=images, padding="max_length", return_tensors="pt",
                                    truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image[:, 0]
                probs = torch.sigmoid(logits)
            results = probs.tolist()

            # Logujemy statystyki rankingu
            res_list = [results] if isinstance(results, float) else results
            logger.info("Ranking zakończony", extra={
                "image_count": len(images),
                "max_score": max(res_list) if res_list else 0
            })

            return res_list
        except Exception as e:
            logger.error("Błąd podczas rankowania obrazów", exc_info=True)
            return [0.0] * len(images)


class AsyncSmartRouter:
    def __init__(self):
        self.cost_calc = CostCalculator()
        self.ranker = LocalVisualRanker()  # Model ładuje się tutaj
        self.llm_model = genai.GenerativeModel('gemini-3-flash-preview')
        # Prosty limiter: serializuje requesty do Gemini; retry z odczekaniem tylko na 429.
        self._gemini_lock = asyncio.Lock()

    async def _gemini_generate(self, *args, **kwargs):
        async with self._gemini_lock:
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    return await self.llm_model.generate_content_async(*args, **kwargs)
                except ResourceExhausted:
                    if attempt >= max_attempts:
                        raise
                    logger.warning(
                        "Gemini quota hit (429), czekam 25s przed ponowieniem",
                        extra={"wait_seconds": 25, "attempt": attempt},
                    )
                    await asyncio.sleep(25)

    async def _fetch_json(self, session, url, params=None, headers=None, source="API"):
        start_time = time.time()
        try:
            async with session.get(url, params=params, headers=headers, timeout=8) as response:
                duration = time.time() - start_time
                if response.status == 200:
                    logger.debug(f"Fetch success: {source}", extra={"url": url, "duration": duration, "source": source})
                    return await response.json()
                else:
                    logger.warning(f"Fetch failed: {source}",
                                   extra={"status_code": response.status, "url": url, "source": source})
        except Exception as e:
            logger.error(f"Fetch error: {source}", exc_info=True, extra={"url": url, "source": source})
        return None

    def _download_image_sync(self, url):
        """
        Pobiera obraz i AUTOMATYCZNIE przycina go do 16:9.
        """
        try:
            resp = requests.get(url, stream=True, timeout=5, headers={"User-Agent": "Bot/1.0"})
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                
                # TUTAJ DZIEJE SIĘ MAGIA:
                # Przycinamy obraz zanim trafi do SigLIPa lub Gemini
                img_cropped = self._crop_to_16_9(img)
                
                return img_cropped
        except Exception as e:
            logger.warning(f"Błąd pobierania/cropowania", extra={"url": url, "error": str(e)})
            return None
        
    def _crop_to_16_9(self, img: Image.Image) -> Image.Image:
        """
        Inteligentne przycinanie (Center Crop) do formatu 16:9.
        Działa na obiekcie PIL.Image.
        """
        original_width, original_height = img.size
        target_ratio = 16 / 9
        current_ratio = original_width / original_height

        if current_ratio == target_ratio:
            return img

        if current_ratio > target_ratio:
            # Obraz jest za szeroki (np. panorama) -> przycinamy boki
            new_width = int(original_height * target_ratio)
            offset = (original_width - new_width) // 2
            # box = (left, upper, right, lower)
            return img.crop((offset, 0, original_width - offset, original_height))
        else:
            # Obraz jest za wysoki (np. 3:2, 4:3) -> przycinamy górę i dół
            new_height = int(original_width / target_ratio)
            offset = (original_height - new_height) // 2
            return img.crop((0, offset, original_width, original_height - offset))

    def _is_wide_enough(self, width: int, height: int) -> bool:
        """
        Sprawdza, czy zdjęcie jest poziome (landscape).
        Akceptujemy 3:2, 4:3, 16:10 itd., bo i tak je przytniemy.
        Odrzucamy tylko pionowe i kwadraty, bo po przycięciu do 16:9 zostałby wąski pasek.
        """
        if not width or not height:
            return False
        # Wymagamy, aby szerokość była przynajmniej 1.2 razy większa od wysokości.
        # To eliminuje kwadraty i piony, ale przepuszcza standardowe zdjęcia z aparatów.
        return width > (height * 1.2)

    # --- API SEARCH (ZAKTUALIZOWANE) ---

    async def search_stock(self, session, query, limit=5):
        tasks = []
        
        # 1. Pexels (orientation=landscape)
        if PEXELS_API_KEY:
            tasks.append(self._fetch_json(
                session, 
                "https://api.pexels.com/v1/search",
                {"query": query, "per_page": limit, "orientation": "landscape"}, 
                {"Authorization": PEXELS_API_KEY},
                source="Pexels"
            ))
            
        # 2. Pixabay (orientation=horizontal)
        if PIXABAY_API_KEY:
            tasks.append(self._fetch_json(
                session, 
                "https://pixabay.com/api/",
                {"key": PIXABAY_API_KEY, "q": query, "image_type": "photo",
                 "safesearch": "true", "per_page": limit, "orientation": "horizontal"}, 
                source="Pixabay"
            ))
            
        # 3. Unsplash (orientation=landscape)
        if UNSPLASH_ACCESS_KEY:
            tasks.append(
                self._fetch_json(
                    session,
                    "https://api.unsplash.com/search/photos",
                    {
                        "query": query,
                        "per_page": limit,
                        "content_filter": "high",
                        "orientation": "landscape",
                        "order_by": "relevant",
                        "lang": "en"
                    },
                    {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
                    source="Unsplash"
                )
            )

        raw_results = []
        responses = await asyncio.gather(*tasks)

        for data in responses:
            if not data: continue
            
            # Pexels
            if 'photos' in data:
                for p in data['photos']:
                    # Używamy nowej luźniejszej funkcji filtrującej
                    if self._is_wide_enough(p.get('width'), p.get('height')):
                        raw_results.append({
                            "source": "Pexels", 
                            "thumb_url": p['src']['large2x'], # Bierzemy większe do cropowania
                            "full_url": p['src']['original'],
                            "id": str(p['id'])
                        })
                        
            # Pixabay
            elif 'hits' in data:
                for h in data['hits']:
                    if self._is_wide_enough(h.get('imageWidth'), h.get('imageHeight')):
                        raw_results.append({
                            "source": "Pixabay", 
                            "thumb_url": h.get('largeImageURL'), # Pixabay webformat może być za mały na dobry crop
                            "full_url": h.get('largeImageURL'), 
                            "id": str(h['id'])
                        })
                        
            # Unsplash
            elif 'results' in data:
                for i in data['results']:
                    if self._is_wide_enough(i.get('width'), i.get('height')):
                        raw_results.append({
                            "source": "Unsplash", 
                            "thumb_url": i['urls']['regular'], # Regular jest wystarczający (~1080px width)
                            "full_url": i['urls']['full'], 
                            "id": i['id']
                        })

        # Deduplikacja po URL miniaturki
        unique = {c['thumb_url']: c for c in raw_results}.values()
        return list(unique)
    # --- MAIN LOGIC ---
    async def process_article(self, text_fragment, *, allow_fallback: bool = True):
        # 1. GENEROWANIE ZAPYTAŃ
        logger.info("Rozpoczynanie analizy artykułu", extra={"text_length": len(text_fragment)})

        prompt = f"""
        Otrzymujesz fragment artykułu:
        "{text_fragment[:500]}..."

        KROK 1:
            Twoim zadaniem jest dobrać słowa kluczowe, które pozwolą przedstawić treść artykułu jednym zdjęciem.
        KROK 2:
            Zwrócona treść, zwłaszcza słowa kluczowe muszą być w języku Angielskim.
        KROK 3:
        Zwróć WYŁĄCZNIE poprawny JSON:
        {{
          "queries": ["str", "str"],
          "visual_subject": "konkretny, namacalny obiekt lub scena (bez abstraktów)"
        }}
        """
        try:
            resp = await self._gemini_generate(
                prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            self.cost_calc.add_gemini_usage(resp.usage_metadata)
            plan = json.loads(resp.text)
        except Exception as e:
            logger.error("Błąd generowania promptów w Gemini", exc_info=True)
            return []

        if isinstance(plan, list):
            if all(isinstance(item, str) for item in plan):
                plan = {"queries": plan}
            else:
                plan = next(
                    (
                        item for item in plan
                        if isinstance(item, dict)
                           and ("queries" in item or "visual_subject" in item)
                    ),
                    {},
                )
        if not isinstance(plan, dict):
            logger.error("Nieprawidłowy format planu", extra={"plan_type": type(plan).__name__})
            return []

        queries = plan.get("queries", [])[:2]
        visual_subject = plan.get("visual_subject", "scene")
        siglip_prompt = f"a photo of {visual_subject}"

        logger.info("Wygenerowano plan", extra={"queries": queries, "siglip_prompt": siglip_prompt})

        # 2. POBIERANIE METADANYCH
        candidates = []
        async with aiohttp.ClientSession() as session:
            tasks = [self.search_stock(session, q) for q in queries]
            groups = await asyncio.gather(*tasks)
            for g in groups: candidates.extend(g)

        # Deduplikacja
        unique = {c['thumb_url']: c for c in candidates if c.get('thumb_url')}.values()
        candidates = list(unique)

        if not candidates:
            logger.warning("Nie znaleziono żadnych zdjęć w API")
            return []

        # 3. RANKING SIGLIP
        to_check = candidates[:15]
        loop = asyncio.get_running_loop()

        # Download images
        futures = [loop.run_in_executor(None, self._download_image_sync, c['thumb_url']) for c in to_check]
        imgs = await asyncio.gather(*futures)

        valid_cands, valid_imgs = [], []
        for c, img in zip(to_check, imgs):
            if img:
                valid_cands.append(c)
                valid_imgs.append(img)

        if not valid_imgs:
            logger.error("Błąd pobierania obrazków do analizy (wszystkie pobierania nieudane)")
            return []

        scores = await loop.run_in_executor(None, self.ranker.rank_images, siglip_prompt, valid_imgs)
        for c, s in zip(valid_cands, scores): c['siglip_score'] = s

        valid_cands.sort(key=lambda x: x['siglip_score'], reverse=True)
        top_cands = valid_cands[:5]

        # 4. GEMINI VISION CHECK
        logger.info("Rozpoczynanie weryfikacji wizualnej Gemini", extra={"candidates_count": len(top_cands)})
        final = []
        for c in top_cands:
            img = self._download_image_sync(c['thumb_url'])
            if not img: continue

            check_prompt = f"""
                        Czy to zdjęcie pasuje do tematu: "{text_fragment[:200]}"?
                        oraz kategorii: {queries}?
                        Odrzuć grafiki, tekst, wektory.
                        JSON: {{ "suitable": true/false, "reason": "..." }}
                        """
            try:
                res = await self._gemini_generate(
                    [check_prompt, img],
                    generation_config={"response_mime_type": "application/json"},
                )
                self.cost_calc.add_gemini_usage(res.usage_metadata)
                verdict = json.loads(res.text)
                c['check'] = verdict
                if verdict.get('suitable'): final.append(c)
            except Exception:
                logger.warning("Błąd weryfikacji Gemini dla jednego zdjęcia", exc_info=True)

        # Fallback
        if allow_fallback and not final and top_cands:
            logger.warning("Brak zaakceptowanych zdjęć przez Gemini, używam fallback (Top 1 SigLIP)")
            top_cands[0]['check'] = {"suitable": True, "reason": "Fallback: SigLIP best match"}
            final.append(top_cands[0])

        # Logowanie kosztów na koniec requestu
        self.cost_calc.log_summary()
        return final


# ==========================================
# CZĘŚĆ 2: SERWIS WEBOWY (STARLETTE)
# ==========================================

@asynccontextmanager
async def lifespan(app):
    logger.info("🚀 Startowanie aplikacji", extra={"event": "startup"})
    try:
        app.state.router = AsyncSmartRouter()
        yield
    except Exception as e:
        logger.critical("Krytyczny błąd podczas startu aplikacji", exc_info=True)
        raise e
    finally:
        logger.info("🛑 Zamykanie aplikacji", extra={"event": "shutdown"})


def has_suitable_result(results: list[dict]) -> bool:
    for result in results:
        if result.get("check", {}).get("suitable") is True and isinstance(
                result.get("siglip_score"), (int, float)
        ):
            return True
    return False


async def analyze_endpoint(request: Request):
    # Generowanie kontekstu requestu dla logów (jeśli OTel nie złapie)
    start_time = time.time()
    try:
        body = await request.json()
        text = body.get("text")

        if not text:
            logger.warning("Otrzymano request bez pola 'text'", extra={"status": 400})
            return JSONResponse({"error": "Brak pola 'text' w JSON"}, status_code=400)

        max_attempts = 3
        results = []
        for attempt in range(1, max_attempts + 1):
            results = await request.app.state.router.process_article(
                text,
                allow_fallback=False,
            )
            if has_suitable_result(results):
                break
            logger.warning(
                "Brak suitable wynikow, ponawiam przetwarzanie",
                extra={"attempt": attempt, "max_attempts": max_attempts},
            )

        duration = time.time() - start_time
        logger.info("Przetworzono request", extra={
            "status": 200,
            "result_count": len(results),
            "duration": round(duration, 3)
        })

        return JSONResponse({
            "status": "success",
            "count": len(results),
            "results": results
        })

    except Exception as e:
        logger.error("Nieobsłużony błąd w endpoincie", exc_info=True, extra={"status": 500})
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def health_check(request):
    # Healthcheck nie musi generować dużej ilości logów, chyba że debug
    logger.debug("Health check probe")
    return JSONResponse({"status": "online"})


routes = [
    Route("/analyze", endpoint=analyze_endpoint, methods=["POST"]),
    Route("/health", endpoint=health_check, methods=["GET"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(debug=False, routes=routes, lifespan=lifespan, middleware=middleware)

if __name__ == "__main__":
    # Konfiguracja uvicorn logging jest nadpisywana przez naszą konfigurację globalną
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
