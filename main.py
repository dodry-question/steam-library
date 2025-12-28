import os
import json
import re
import random
import asyncio
import httpx
from dotenv import load_dotenv
load_dotenv(override=True)
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from urllib.parse import quote, unquote

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Field, Session, SQLModel, create_engine, select

# --- НАСТРОЙКИ ---
# Загружаем и очищаем ключ от лишних пробелов и кавычек
RAW_KEY = os.environ.get("STEAM_API_KEY") or ""
STEAM_API_KEY = RAW_KEY.strip().replace('"', '').replace("'", "")

if not STEAM_API_KEY:
    print("❌ ОШИБКА: Ключ Steam не найден!")
else:
    print(f"✅ Ключ загружен и очищен: {STEAM_API_KEY[:5]}***")
MY_DOMAIN = os.environ.get("MY_DOMAIN", "http://localhost:8001")
STORE_API_LOCK = asyncio.Lock()

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
    game_names: Dict[int, str] 

sqlite_file_name = "games.db"
connect_args = {"check_same_thread": False}
engine = create_engine(f"sqlite:///{sqlite_file_name}", connect_args=connect_args)

def create_db_and_tables():
    try:
        SQLModel.metadata.create_all(engine)
    except Exception as e:
        print(f"⚠️ Ошибка создания БД: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def on_startup():
    create_db_and_tables()

# --- Вспомогательные функции ---

async def request_store(client, app_id, region="ru"):
    url = "https://store.steampowered.com/api/appdetails"
    params = {
        "appids": str(app_id),
        "cc": region,
        "l": "russian"
    }
    # Оставляем только один заголовок, самый простой
    headers = {"User-Agent": "Mozilla/5.0"} 

    try:
        resp = await client.get(url, params=params, headers=headers, timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            print(f"🛑 429: Блокировка за частоту. Ждем 10 секунд...")
            await asyncio.sleep(10)
        return None
    except Exception:
        return None

async def fetch_steam_store_data(client: httpx.AsyncClient, app_ids: List[int]):
    if not app_ids: return {}
    target_id = app_ids[0]
    sid_str = str(target_id)

    # Запрашиваем RU данные
    data = await request_store(client, target_id, region="ru")
    
    # Если RU не ответил или вернул False (игра недоступна в РФ)
    if not data or not data.get(sid_str, {}).get('success'):
        await asyncio.sleep(0.5) # Маленькая пауза
        # Пытаемся получить данные из США (там цены есть почти всегда)
        data = await request_store(client, target_id, region="us")
        
    return data if data else {}
    
def parse_game_obj(steam_id: int, data: dict, known_name: str) -> Game:
    image_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{steam_id}/header.jpg"
    success = data.get('success', False)
    game_data = data.get('data', {})

    name = game_data.get('name', known_name)
    # Исправляем логику: если success=True, но цены нет - значит игра бесплатная или это спец. версия
    price_str = "В библиотеке" # По умолчанию для твоих игр
    discount = 0

    if success:
        if game_data.get('is_free'):
            price_str = "Бесплатно"
        elif 'price_overview' in game_data:
            p = game_data['price_overview']
            price_str = p.get('final_formatted', "Бесплатно")
            discount = p.get('discount_percent', 0)
    
    return Game(
        steam_id=steam_id,
        name=name,
        image_url=image_url,
        price_str=price_str,
        discount_percent=discount,
        last_updated=datetime.now()
    )

# --- API ---

@app.post("/api/games-batch")
async def get_games_batch(payload: BatchRequest):
    return StreamingResponse(game_generator(payload), media_type="application/x-ndjson")

async def game_generator(payload: BatchRequest):
    try:
        ids = payload.steam_ids
        playtimes = payload.playtimes
        names_map = payload.game_names 
        
        ids_to_fetch = []
        cutoff = datetime.now() - timedelta(hours=12) 

        with Session(engine) as session:
            try:
                existing_games = session.exec(select(Game).where(Game.steam_id.in_(ids))).all()
                existing_map = {g.steam_id: g for g in existing_games}
            except:
                existing_map = {}

            for steam_id in ids:
                game = existing_map.get(steam_id)
                # Если игра есть в БД и данные свежие
                if game and game.last_updated > cutoff:
                    d = game.model_dump()
                    if d.get('last_updated'): d['last_updated'] = d['last_updated'].isoformat()
                    d['playtime_forever'] = playtimes.get(steam_id, 0)
                    yield json.dumps(d, ensure_ascii=False) + "\n"
                else:
                    ids_to_fetch.append(steam_id)

        if not ids_to_fetch:
            return

        # Размер пачки 15 - оптимально для стабильности
        CHUNK_SIZE = 1
        chunks = [ids_to_fetch[i:i + CHUNK_SIZE] for i in range(0, len(ids_to_fetch), CHUNK_SIZE)]

        async with httpx.AsyncClient() as client:
            for chunk in chunks:
                store_resp = await fetch_steam_store_data(client, chunk)
                games_to_save = []
                
                for sid in chunk:
                    sid_str = str(sid)
                    data = store_resp.get(sid_str, {})
                    known_name = names_map.get(sid, f"App {sid}")
                    
                    game_obj = parse_game_obj(sid, data, known_name)
                    games_to_save.append(game_obj)

                if games_to_save:
                    try:
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
                                
                                if d.get('last_updated'): d['last_updated'] = d['last_updated'].isoformat()
                                d['playtime_forever'] = playtimes.get(g.steam_id, 0)
                                yield json.dumps(d, ensure_ascii=False) + "\n"
                            session.commit()
                    except Exception as e:
                        print(f"Ошибка БД: {e}")
                        for g in games_to_save:
                            d = g.model_dump()
                            if isinstance(d.get('last_updated'), datetime): d['last_updated'] = d['last_updated'].isoformat()
                            d['playtime_forever'] = playtimes.get(g.steam_id, 0)
                            yield json.dumps(d, ensure_ascii=False) + "\n"

    except Exception as e:
        print(f"❌ Generator Error: {e}")

@app.get("/api/get-games-list")
async def get_games_list(request: Request, user_id: Optional[str] = None):
    target_id = None
    if user_id:
        target_id = await resolve_steam_id(user_id)
    if not target_id:
        target_id = request.cookies.get("user_steam_id")
    
    if not target_id:
        return {"error": "User ID not found"}

    url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={STEAM_API_KEY}&steamid={target_id}&format=json&include_appinfo=1&include_played_free_games=1"
    
    async with httpx.AsyncClient() as client:
        try:
            p_name = target_id
            try:
                user_url = f"https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/?key={STEAM_API_KEY}&steamids={target_id}"
                u_resp = await client.get(user_url, timeout=5.0)
                u_data = u_resp.json()
                if 'response' in u_data and 'players' in u_data['response'] and u_data['response']['players']:
                    p_name = u_data['response']['players'][0]['personaname']
            except: pass 

            resp = await client.get(url, timeout=20.0)
            if resp.status_code == 403: return {"error": "Steam API Key Error (403)"}
            if resp.status_code != 200: return {"error": f"Steam API Error: {resp.status_code}"}

            data = resp.json()
            if "response" in data and "games" in data["response"]:
                games = []
                for g in data["response"]["games"]:
                    games.append({
                        "appid": g["appid"], 
                        "name": g.get("name", f"App {g['appid']}"),
                        "playtime_forever": g.get("playtime_forever", 0) # Исправили ключ
                    })
                return {"target_id": target_id, "target_name": p_name, "games": games}
            else:
                return {"error": "Профиль скрыт или игр нет"}
        except Exception as e:
            return {"error": f"Server Error: {str(e)}"}

async def resolve_steam_id(input_str: str) -> Optional[str]:
    input_str = input_str.strip()
    if input_str.isdigit() and len(input_str) == 17: return input_str
    clean = input_str.split('/')[-1] if '/' not in input_str else input_str.rstrip('/').split('/')[-1]
    url = f"https://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/?key={STEAM_API_KEY}&vanityurl={clean}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url)
            d = resp.json()
            if d['response']['success'] == 1: return d['response']['steamid']
        except: pass
    return None

@app.post("/api/add-game")
async def add_game_manual(steam_id: int = Form(...)):
    payload = BatchRequest(steam_ids=[steam_id], playtimes={steam_id: 0}, game_names={})
    async for item in game_generator(payload):
        return json.loads(item)
    return {"error": "Не удалось загрузить"}

# --- ИИ ---
@app.post("/api/recommend")
async def recommend(request: Request):
    try:
        body = await request.json()
        games = body.get("games", [])
        
        # 1. Берем игры > 5 часов (300 мин). Ключ playtime_forever
        liked_games = [g for g in games if g.get('playtime_forever', 0) > 300]
        
        # Fallback
        if not liked_games:
            liked_games = sorted(games, key=lambda x: x.get('playtime_forever', 0), reverse=True)[:20]

        selection = random.sample(liked_games, min(3, len(liked_games)))
        names = ", ".join([g['name'] for g in selection])
        
        prompt = (
            f"User likes: {names}. Suggest 3 similar Steam games. "
            f"STRICT FORMAT REQUIRED: ID: <appid> | Name: <name> | Reason: <short russian text>. "
            f"IMPORTANT: Use REAL and ACCURATE Steam AppIDs. "
            f"DO NOT write introductory text. DO NOT ask questions. JUST THE LIST."
        )
        print(f"🤖 AI Request (Selected): {names}")

        async with httpx.AsyncClient() as client:
            resp = await client.post("https://text.pollinations.ai/", json={
                "messages": [{"role": "user", "content": prompt}],
                "model": "openai",
                "seed": random.randint(1, 9999)
            }, timeout=45.0)
            
            text = resp.text
            print(f"🤖 AI Response: {text}")

            recs = []
            for line in text.split('\n'):
                if "ID:" in line:
                    try:
                        parts = line.split("|")
                        if len(parts) >= 3:
                            app_id = int(re.search(r'\d+', parts[0]).group())
                            recs.append({
                                "steam_id": app_id,
                                "name": parts[1].split(":")[1].strip(),
                                "ai_reason": parts[2].split(":")[1].strip(),
                                "image_url": f"https://cdn.akamai.steamstatic.com/steam/apps/{app_id}/header.jpg",
                                "genres": "AI Recommended",
                                "price_str": "?",
                                "discount_percent": 0
                            })
                    except: pass
            return {"content": {"recommendations": recs}}
    except Exception as e:
        return {"content": {"error": str(e)}}

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
        user_name = "Steam User"
        user_avatar = ""
        
        # Запрос данных профиля
        api_url = f"https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/?key={STEAM_API_KEY}&steamids={sid}"
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(api_url, timeout=10.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('response', {}).get('players'):
                        player = data['response']['players'][0]
                        user_name = player.get('personaname', 'Steam User')
                        # Берем аватарку покрупнее для красоты
                        user_avatar = player.get('avatarfull', '') or player.get('avatarmedium', '')
                else:
                    print(f"❌ Steam API Profile Error: {resp.status_code}")
            except Exception as e:
                print(f"❌ Auth Error: {e}")

        resp = RedirectResponse("/")
        resp.set_cookie("user_steam_id", sid)
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
    uname = unquote(request.cookies.get("user_name") or "")
    uavatar = request.cookies.get("user_avatar")
    return templates.TemplateResponse("index.html", {
        "request": request, "user_id": uid, "user_name": uname, "user_avatar": uavatar
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, proxy_headers=True, forwarded_allow_ips="*")