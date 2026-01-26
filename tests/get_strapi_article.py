import os

import aiohttp
import asyncio
import json

# ============================================================
# GOOGLE SHEETS
# ============================================================
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1OGqGTXo16-dGZDv-EH2MjJfeIpDOT3TGE7JSUh193YE"
SHEET_NAME = "Arkusz1"


def get_sheets_service():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("sheets", "v4", credentials=creds)


def append_row(
    title: str,
    lead: str,
    category: str = "",
    portal: str = "",
    source: str = "",
    image_url: str = "",
    score: float | None = None,
    reason: str = "",
    status: str = "OK",
):
    service = get_sheets_service()

    body = {
        "values": [[
            title,
            lead,
            category,
            portal,
            source,
            image_url,
            score,
            reason,
            status,
        ]]
    }

    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A:I",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()

# =========================
# CONFIG
# =========================

portal_list = [
    "RolnikINFO",
    "BiznesINFO",
    "Pacjenci",
]

STRAPI_URL = "https://strapi.prod.iberion.net/api/articles"
CONTENT_BRIEF_URL = "http://46.62.231.49:31501/api/generate-brief"

strapi_token = os.getenv("STRAPI_TOKEN")

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {strapi_token}",
}

# =========================
# STRAPI
# =========================

async def fetch_articles_for_portal(session, portal_name):
    params = {
        "filters[portal][PortalName][$eq]": portal_name,
        "sort": "updatedAt:desc",
        "pagination[pageSize]": 10,   # tylko jeden (najnowszy)
        "populate": "*",
    }

    async with session.get(
        STRAPI_URL,
        headers=headers,
        params=params,
        timeout=30,
    ) as resp:
        resp.raise_for_status()
        return await resp.json()

# =========================
# ARTICLE CONTENT
# =========================

def build_article_content(attrs: dict) -> str:
    parts = []

    def add(value):
        if value and isinstance(value, str):
            parts.append(value.strip())

    add(attrs.get("title"))
    add(attrs.get("lead"))

    for i in range(1, 5):
        add(attrs.get(f"body{i}Title"))
        add(attrs.get(f"body{i}"))

    return "\n\n".join(parts)

def extract_article_fields(api_response):
    data = api_response.get("data", [])
    if not data:
        return None

    attrs = data[0].get("attributes", {})

    return {
        "title": attrs.get("title"),
        "lead": attrs.get("lead"),
        "content": build_article_content(attrs),
    }

# =========================
# CONTENT BRIEF GENERATOR
# =========================

async def get_content_brief(session, content: str):
    payload = {
        "content": content,
    }

    async with session.post(
        CONTENT_BRIEF_URL,
        json=payload,
        timeout=30,
    ) as resp:
        resp.raise_for_status()
        return await resp.json()

def extract_category(brief_response):
    """
    Obsługiwane formaty:
    1) { "category": "..." }
    2) [ { "category": "..." } ]
    """

    # case: dict
    if isinstance(brief_response, dict):
        return brief_response.get("category")

    # case: list[dict]
    if isinstance(brief_response, list) and brief_response:
        first = brief_response[0]
        if isinstance(first, dict):
            return first.get("category")

    return None


# =========================
# MAIN PIPELINE
# =========================

async def process_portal(session, portal_name):
    api_response = await fetch_articles_for_portal(session, portal_name)
    article = extract_article_fields(api_response)

    if not article:
        return None

    brief_response = await get_content_brief(session, article["content"])
    category = extract_category(brief_response)

    return {
        "portal": portal_name,
        "title": article["title"],
        "lead": article["lead"],
        "category": category,
    }

async def analyze_text(session: aiohttp.ClientSession, text: str, category: str):
    async with session.post(
        "http://127.0.0.1:8000/analyze",
        json={"text": text, "category": category},
        timeout=120,
    ) as response:
        return response.status, await response.text()

def build_analyze_text(title: str, lead: str | None = None) -> str:
    if lead:
        return f"{title}\n{lead}"
    return title

async def main():
    results = []

    async with aiohttp.ClientSession() as session:
        for portal in portal_list:
            try:
                result = await process_portal(session, portal)
                if result:
                    text = build_analyze_text(
                        title=result.get("title") or "",
                        lead=result.get("lead"),
                    )
                    status_code, body = await analyze_text(
                        session,
                        text=text,
                        category=result.get("category") or "",
                    )
                    status = "OK" if status_code == 200 else "ERROR"
                    category = result.get("category") or ""
                    source = ""
                    image_url = ""
                    score = None
                    reason = category
                    if status == "OK":
                        try:
                            parsed = json.loads(body)
                            results = parsed.get("results") or []
                            if not results:
                                status = "ERROR"
                                reason = f"{category} | Empty results from analyze".strip(" |")
                            else:
                                top = results[0]
                                source = top.get("source") or source
                                image_url = top.get("full_url") or ""
                                score = top.get("siglip_score")
                                reason = top.get("check", {}).get("reason") or reason
                        except Exception:
                            status = "ERROR"
                            reason = f"{category} | Invalid JSON from analyze".strip(" |")
                    if status != "OK":
                        reason = f"{reason} | Analyze error: {status_code}".strip(" |")
                    append_row(
                        title=result.get("title") or "",
                        lead=result.get("lead") or "",
                        category=category,
                        portal=portal,
                        source=source,
                        image_url=image_url,
                        score=score,
                        reason=reason,
                        status=status,
                    )
                    results.append(result)
            except Exception as e:
                print(f"❌ Błąd dla {portal}: {e}")

    print(json.dumps(results, indent=2, ensure_ascii=False))

# =========================
# ENTRYPOINT
# =========================

if __name__ == "__main__":
    asyncio.run(main())
