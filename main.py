import os
import json
import re
import random
import asyncio
import httpx
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from urllib.parse import quote, unquote

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlmodel import Field, Session, SQLModel, create_engine, select

# --- НАСТРОЙКИ ---
STEAM_API_KEY = os.environ.get("STEAM_API_KEY") 
MY_DOMAIN = os.environ.get("MY_DOMAIN", "http://localhost:8000")

# Глобальная блокировка для запросов к Store API
STORE_API_LOCK = asyncio.Lock()

# Курсы валют
RATE_KZT_TO_RUB = 0.21  
RATE_USD_TO_RUB = 95.0 

# --- База данных ---
class Game(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    steam_id: int = Field(index=True, unique=True)
    name: str
    image_url: str
    genres: str | None = None
    price_str: str | None = None 
    discount_percent: int = 0
    last_updated: datetime = Field(default_factory=datetime.now)

class BatchRequest(SQLModel):
    steam_ids: List[int]
    playtimes: Dict[int, int]

sqlite_file_name = "games.db"
engine = create_engine(f"sqlite:///{sqlite_file_name}")

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

app = FastAPI()
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def on_startup():
    create_db_and_tables()
    asyncio.create_task(update_currency_rates())

async def update_currency_rates():
    global RATE_KZT_TO_RUB, RATE_USD_TO_RUB
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://www.cbr-xml-daily.ru/daily_json.js")
            data = resp.json()
            RATE_USD_TO_RUB = data["Valute"]["USD"]["Value"]
            kzt = data["Valute"]["KZT"]
            RATE_KZT_TO_RUB = kzt["Value"] / kzt["Nominal"]
            print(f"💱 Курсы обновлены: USD={RATE_USD_TO_RUB:.2f}, KZT={RATE_KZT_TO_RUB:.4f}")
    except:
        print("⚠️ Не удалось обновить курсы, используем стандартные.")

# --- Вспомогательные функции ---

async def fetch_steam_store_data(client: httpx.AsyncClient, app_ids: List[int], region: str):
    if not app_ids:
        return {}
    
    ids_str = ",".join(map(str, app_ids))
    url = "https://store.steampowered.com/api/appdetails"
    params = {
        "appids": ids_str,
        "l": "russian",
        "cc": region,
        "filters": "price_overview,basic,genres"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    async with STORE_API_LOCK:
        try:
            # Важный sleep, чтобы не получать баны
            await asyncio.sleep(1.6)
            resp = await client.get(url, params=params, headers=headers, timeout=20.0)
            
            if resp.status_code == 429:
                print("🛑 429 Rate Limit! Ждем 60 секунд...")
                await asyncio.sleep(60) 
                return await fetch_steam_store_data(client, app_ids, region)
            
            if resp.status_code == 200:
                return resp.json()
                
        except Exception as e:
            print(f"❌ Ошибка запроса к Steam Store: {e}")
            
    return {}

def parse_game_obj(steam_id: int, data: dict, region: str) -> Game:
    success = data.get('success', False)
    if not success:
        return Game(steam_id=steam_id, name=f"App {steam_id}", image_url="", price_str="Недоступно", discount_percent=0)

    game_data = data.get('data', {})
    name = game_data.get('name', f"App {steam_id}")
    image = game_data.get('header_image', '')
    
    genres = []
    for g in game_data.get('genres', []):
        genres.append(g['description'])
    genres_str = ", ".join(genres) if genres else "N/A"

    price_str = "Не продается"
    discount = 0

    if game_data.get('is_free'):
        price_str = "Бесплатно"
    elif 'price_overview' in game_data:
        p = game_data['price_overview']
        discount = p.get('discount_percent', 0)
        final = p.get('final', 0)
        
        if region == 'kz':
            rub = int((final / 100) * RATE_KZT_TO_RUB)
            price_str = f"~{rub} ₽"
        elif region == 'us':
            rub = int((final / 100) * RATE_USD_TO_RUB)
            price_str = f"~{rub} ₽"
        else:
            price_str = p.get('final_formatted', f"{final/100}")

    return Game(
        steam_id=steam_id,
        name=name,
        image_url=image,
        genres=genres_str,
        price_str=price_str,
        discount_percent=discount,
        last_updated=datetime.now()
    )

# --- API Endpoints ---

@app.post("/api/games-batch")
async def get_games_batch(payload: BatchRequest):
    return StreamingResponse(game_generator(payload), media_type="application/x-ndjson")

async def game_generator(payload: BatchRequest):
    ids = payload.steam_ids
    playtimes = payload.playtimes
    ids_to_fetch = []
    cutoff = datetime.now() - timedelta(days=3)

    with Session(engine) as session:
        existing_games = session.exec(select(Game).where(Game.steam_id.in_(ids))).all()
        existing_map = {g.steam_id: g for g in existing_games}

        for steam_id in ids:
            game = existing_map.get(steam_id)
            if game and game.last_updated > cutoff:
                d = game.model_dump()
                d['playtime_forever'] = playtimes.get(steam_id, 0)
                yield json.dumps(d, ensure_ascii=False) + "\n"
            else:
                ids_to_fetch.append(steam_id)

    if not ids_to_fetch:
        return

    CHUNK_SIZE = 25 
    chunks = [ids_to_fetch[i:i + CHUNK_SIZE] for i in range(0, len(ids_to_fetch), CHUNK_SIZE)]

    async with httpx.AsyncClient() as client:
        for chunk in chunks:
            # 1. RU
            ru_resp = await fetch_steam_store_data(client, chunk, 'ru')
            games_to_save = []
            kz_needed = []

            for sid in chunk:
                sid_str = str(sid)
                data = ru_resp.get(sid_str, {})
                if data.get('success'):
                    game_obj = parse_game_obj(sid, data, 'ru')
                    if game_obj.price_str != "Не продается":
                        games_to_save.append(game_obj)
                    else:
                        kz_needed.append(sid)
                else:
                    kz_needed.append(sid)

            # 2. KZ
            if kz_needed:
                kz_resp = await fetch_steam_store_data(client, kz_needed, 'kz')
                for sid in kz_needed:
                    sid_str = str(sid)
                    data = kz_resp.get(sid_str, {})
                    game_obj = parse_game_obj(sid, data, 'kz')
                    games_to_save.append(game_obj)

            # Сохранение и отдача
            if games_to_save:
                with Session(engine) as session:
                    for g in games_to_save:
                        existing = session.exec(select(Game).where(Game.steam_id == g.steam_id)).first()
                        if existing:
                            existing.name = g.name
                            existing.image_url = g.image_url
                            existing.genres = g.genres
                            existing.price_str = g.price_str
                            existing.discount_percent = g.discount_percent
                            existing.last_updated = datetime.now()
                            session.add(existing)
                            d = existing.model_dump()
                        else:
                            session.add(g)
                            d = g.model_dump()
                        
                        d['playtime_forever'] = playtimes.get(g.steam_id, 0)
                        yield json.dumps(d, ensure_ascii=False) + "\n"
                    session.commit()

@app.get("/api/get-games-list")
async def get_games_list(request: Request, user_id: Optional[str] = None):
    target_id = None
    if user_id:
        target_id = await resolve_steam_id(user_id)
    if not target_id:
        target_id = request.cookies.get("user_steam_id")
    
    if not target_id:
        return {"error": "User ID not found"}

    url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={STEAM_API_KEY}&steamid={target_id}&format=json&include_appinfo=1&include_played_free_games=1"
    
    async with httpx.AsyncClient() as client:
        try:
            # Пытаемся получить имя, но не падаем, если ошибка
            p_name = target_id
            try:
                user_url = f"http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/?key={STEAM_API_KEY}&steamids={target_id}"
                u_resp = await client.get(user_url, timeout=5.0)
                u_data = u_resp.json()
                if 'response' in u_data and 'players' in u_data['response'] and u_data['response']['players']:
                    p_name = u_data['response']['players'][0]['personaname']
            except:
                pass 

            # Игры
            resp = await client.get(url, timeout=10.0)
            if resp.status_code != 200:
                return {"error": f"Steam Error: {resp.status_code}"}

            data = resp.json()
            if "response" in data and "games" in data["response"]:
                games = [{"appid": g["appid"], "playtime": g.get("playtime_forever", 0)} for g in data["response"]["games"]]
                return {"target_id": target_id, "target_name": p_name, "games": games}
            else:
                return {"error": "Профиль скрыт или нет игр"}
        except Exception as e:
            print(f"Error in get_games_list: {e}")
            return {"error": "Ошибка сети или Steam API"}

async def resolve_steam_id(input_str: str) -> Optional[str]:
    input_str = input_str.strip()
    if input_str.isdigit() and len(input_str) == 17:
        return input_str
    
    clean = input_str.split('/')[-1] if '/' not in input_str else input_str.rstrip('/').split('/')[-1]
    url = f"http://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/?key={STEAM_API_KEY}&vanityurl={clean}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url)
            d = resp.json()
            if d['response']['success'] == 1:
                return d['response']['steamid']
        except: pass
    return None

@app.post("/api/add-game")
async def add_game_manual(steam_id: int = Form(...)):
    payload = BatchRequest(steam_ids=[steam_id], playtimes={steam_id: 0})
    async for item in game_generator(payload):
        return json.loads(item)
    return {"error": "Не удалось загрузить"}

@app.post("/api/recommend")
async def recommend(request: Request):
    try:
        body = await request.json()
        games = body.get("games", [])
        top = sorted(games, key=lambda x: x.get('playtime', 0), reverse=True)[:10]
        names = ", ".join([g['name'] for g in top])
        
        prompt = f"Based on games: {names}. Recommend 3 similar games available on Steam. Format strictly: ID: <appid> | Name: <name> | Reason: <short reason>"
        
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://text.pollinations.ai/", json={
                "messages": [{"role": "user", "content": prompt}],
                "model": "openai"
            }, timeout=30.0)
            text = resp.text
            
            recs = []
            for line in text.split('\n'):
                if "ID:" in line:
                    try:
                        parts = line.split("|")
                        if len(parts) >= 3:
                            app_id = int(re.search(r'\d+', parts[0]).group())
                            rec_game = {
                                "steam_id": app_id,
                                "name": parts[1].split(":")[1].strip(),
                                "ai_reason": parts[2].split(":")[1].strip(),
                                "image_url": f"https://cdn.akamai.steamstatic.com/steam/apps/{app_id}/header.jpg",
                                "genres": "AI Recommended",
                                "price_str": "?",
                                "discount_percent": 0
                            }
                            recs.append(rec_game)
                    except: pass
            return {"content": {"recommendations": recs}}
    except Exception as e:
        return {"content": {"error": str(e)}}

# --- Auth System (ИСПРАВЛЕНО) ---
@app.get("/login")
def login():
    params = {
        "openid.ns": "http://specs.openid.net/auth/2.0",
        "openid.mode": "checkid_setup",
        "openid.return_to": f"{MY_DOMAIN}/auth",
        "openid.realm": f"{MY_DOMAIN}",
        "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
        "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
    }
    q = "&".join([f"{k}={v}" for k, v in params.items()])
    return RedirectResponse(f"https://steamcommunity.com/openid/login?{q}")

@app.get("/auth")
async def auth(request: Request):
    params = request.query_params
    if "openid.identity" in params:
        sid = params["openid.identity"].split("/")[-1]
        
        # Получаем инфо о пользователе для куки
        user_name = "Steam User"
        user_avatar = ""
        try:
            api_url = f"http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/?key={STEAM_API_KEY}&steamids={sid}"
            async with httpx.AsyncClient() as client:
                resp = await client.get(api_url)
                data = resp.json()
                player = data['response']['players'][0]
                user_name = player.get('personaname', 'Steam User')
                user_avatar = player.get('avatarmedium', '') or player.get('avatarfull', '')
        except Exception as e:
            print(f"Auth info fetch error: {e}")

        resp = RedirectResponse("/")
        resp.set_cookie("user_steam_id", sid)
        # Кодируем имя, чтобы не было ошибок с русскими буквами
        resp.set_cookie("user_name", quote(user_name))
        resp.set_cookie("user_avatar", user_avatar)
        return resp
        
    return RedirectResponse("/")

@app.get("/logout")
def logout():
    r = RedirectResponse("/")
    r.delete_cookie("user_steam_id")
    r.delete_cookie("user_name")
    r.delete_cookie("user_avatar")
    return r

@app.get("/")
def index(request: Request):
    uid = request.cookies.get("user_steam_id")
    uname = request.cookies.get("user_name")
    uavatar = request.cookies.get("user_avatar")
    
    if uname:
        try:
            uname = unquote(uname)
        except: pass
        
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "user_id": uid,
        "user_name": uname,
        "user_avatar": uavatar
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)