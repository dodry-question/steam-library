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
    
async def search_steam_game(client: httpx.AsyncClient, name: str) -> Optional[int]:
    search_url = "https://store.steampowered.com/api/storesearch/"
    params = {"term": name, "l": "russian", "cc": "ru"}
    try:
        resp = await client.get(search_url, params=params, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("total") > 0:
                return data["items"][0]["id"]
    except: pass
    return None

async def fetch_steam_store_data(client: httpx.AsyncClient, app_ids: List[int]):
    if not app_ids: return {}, False
    
    # Объединяем ID в строку через запятую для пачки
    ids_str = ",".join(map(str, app_ids))
    
    async with STORE_API_LOCK:
        # Пробуем RU регион
        data = await request_store(client, ids_str, region="ru")
        is_fallback = False
        
        # Если Steam не отдал данные (например, регион-лок), пробуем US для всей пачки
        if not data or not any(data.get(str(sid), {}).get('success') for sid in app_ids):
            await asyncio.sleep(1.5) # Пауза перед вторым шансом
            data = await request_store(client, ids_str, region="us")
            is_fallback = True
        
        return (data if data is not None else {}), is_fallback

# Вспомогательная функция (обновите ее, чтобы принимала строку)
async def request_store(client, ids_str, region="ru"):
    url = "https://store.steampowered.com/api/appdetails"
    params = {"appids": ids_str, "cc": region, "l": "russian"}
    headers = {"User-Agent": "Mozilla/5.0"} 

    try:
        resp = await client.get(url, params=params, headers=headers, timeout=15.0)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            print(f"🛑 429: Блокировка. Ждем паузу...")
            return None
    except: return None
    
def parse_game_obj(steam_id: int, data: dict, known_name: str, is_fallback: bool = False) -> Game:
    image_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{steam_id}/header.jpg"
    success = data.get('success', False)
    game_data = data.get('data', {})

    name = game_data.get('name', known_name)
    genres_list = [g['description'] for g in game_data.get('genres', [])]
    genres_str = ", ".join(genres_list) if genres_list else ""

    # Логика определения цены
    price_val = "Нет в продаже"
    discount = 0

    if success:
        if game_data.get('is_free'):
            price_val = "Бесплатно"
        elif 'price_overview' in game_data:
            p = game_data['price_overview']
            price_val = p.get('final_formatted', "Бесплатно")
            discount = p.get('discount_percent', 0)
    
    # Если это fallback (данные из США), добавляем спец. маркер
    if is_fallback and success and price_val not in ["Нет в продаже", "Бесплатно"]:
        final_price = f"LOCKED|{price_val}"
    else:
        final_price = price_val

    return Game(
        steam_id=steam_id,
        name=name,
        image_url=image_url,
        genres=genres_str,
        price_str=final_price,
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
        
        # СОРТИРОВКА: сначала те, в которые больше всего играли
        ids.sort(key=lambda x: playtimes.get(x, 0), reverse=True)
        
        ids_to_fetch = []
        cutoff = datetime.now() - timedelta(hours=12) 

        with Session(engine) as session:
            # Сначала быстро отдаем то, что уже есть в базе
            existing_games = session.exec(select(Game).where(Game.steam_id.in_(ids))).all()
            existing_map = {g.steam_id: g for g in existing_games}

            for steam_id in ids:
                game = existing_map.get(steam_id)
                if game and game.last_updated > cutoff:
                    d = game.model_dump()
                    if d.get('last_updated'): d['last_updated'] = d['last_updated'].isoformat()
                    d['playtime_forever'] = playtimes.get(steam_id, 0)
                    yield json.dumps(d, ensure_ascii=False) + "\n"
                else:
                    ids_to_fetch.append(steam_id)

        if not ids_to_fetch: return

        # Размер пачки увеличиваем до 10. Это в 10 раз ускорит процесс!
        CHUNK_SIZE = 10 
        chunks = [ids_to_fetch[i:i + CHUNK_SIZE] for i in range(0, len(ids_to_fetch), CHUNK_SIZE)]

        async with httpx.AsyncClient() as client:
            for i, chunk in enumerate(chunks):
                store_resp, is_fallback = await fetch_steam_store_data(client, chunk)
                
                if not store_resp:
                    # Если словили бан, ждем и пробуем еще раз один раз
                    await asyncio.sleep(15)
                    store_resp, is_fallback = await fetch_steam_store_data(client, chunk)
                    if not store_resp: continue

                games_to_save = []
                for sid in chunk:
                    sid_str = str(sid)
                    data = store_resp.get(sid_str, {})
                    game_obj = parse_game_obj(sid, data, names_map.get(sid, ""), is_fallback)
                    games_to_save.append(game_obj)

                # Сохраняем пачку в базу и отправляем на фронтенд
                if games_to_save:
                    with Session(engine) as session:
                        for g in games_to_save:
                            existing = session.exec(select(Game).where(Game.steam_id == g.steam_id)).first()
                            if existing:
                                # Обновляем старую запись
                                for key, value in g.model_dump(exclude={"id", "steam_id"}).items():
                                    setattr(existing, key, value)
                                session.add(existing)
                                d = existing.model_dump()
                            else:
                                session.add(g)
                                d = g.model_dump()
                            
                            if d.get('last_updated'): d['last_updated'] = d['last_updated'].isoformat()
                            d['playtime_forever'] = playtimes.get(g.steam_id, 0)
                            yield json.dumps(d, ensure_ascii=False) + "\n"
                        session.commit()
                
                # ВАЖНО: Маленькая пауза 0.8 сек между пачками, чтобы Steam нас не забанил
                # Это незаметно для пользователя, так как данные идут пачками по 10 штук
                await asyncio.sleep(0.8)

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
        all_games = body.get("games", [])
        mood = body.get("mood", "hidden gems") # Получаем настроение
        
        # 1. Формируем "Ядро интересов" - ТОП-10 самых играемых игр
        top_played = sorted(all_games, key=lambda x: x.get('playtime_forever', 0), reverse=True)[:10]
        core_names = ", ".join([g['name'] for g in top_played])
        
        # 2. Формируем список исключений (чтобы не предлагал то, что есть)
        owned_names = ", ".join([g['name'] for g in all_games[:50]]) 

        # 3. Улучшенный промпт
        prompt = (
            f"Ты — опытный игровой куратор. Анализируй библиотеку игрока: {core_names}. "
            f"Найди 3 игры в Steam для категории '{mood}'. "
            f"ПРАВИЛА: "
            f"1. Для каждой рекомендации выбери ОДНУ конкретную игру из списка игрока ({core_names}), на основе которой ты даешь совет. "
            f"2. Избегай очевидных хитов из топ-50 Steam. "
            f"3. НЕ ПРЕДЛАГАЙ игры, которые уже есть: {owned_names}. "
            f"4. ФОРМАТ ОТВЕТА (строго): "
            f"Name: <название> | Based on: <название игры из списка игрока> | Reason: <почему подходит> "
            f"Не пиши ничего, кроме этих строк."
        )

        async with httpx.AsyncClient() as client:
            resp = await client.post("https://text.pollinations.ai/", json={
                "messages": [{"role": "system", "content": "You are a professional game curator."},
                             {"role": "user", "content": prompt}],
                "model": "openai",
                "seed": random.randint(1, 99999)
            }, timeout=45.0)
            
            text = resp.text
            recs = []
            
            # Разбираем ответ построчно
            for line in text.split('\n'):
                line = line.strip()
                if "|" in line and "Based on:" in line:
                    try:
                        parts = line.split("|")
                        if len(parts) < 3: continue
                        
                        g_name = parts[0].replace("Name:", "").strip()
                        g_name = re.sub(r'^\d+\.\s*', '', g_name) # Убираем цифры в начале
                        
                        based_on = parts[1].replace("Based on:", "").strip()
                        reason = parts[2].replace("Reason:", "").strip()

                        # Ищем ID игры в Steam
                        real_id = await search_steam_game(client, g_name)

                        if real_id:
                            recs.append({
                                "steam_id": real_id,
                                "name": g_name,
                                "based_on": based_on,
                                "ai_reason": reason,
                                "image_url": f"https://cdn.akamai.steamstatic.com/steam/apps/{real_id}/header.jpg"
                            })
                    except Exception:
                        continue # Если одна строка битая, идем к следующей
            
            return {"content": {"recommendations": recs}}

    except Exception as e:
        print(f"❌ AI Error: {e}")
        return {"content": {"error": str(e), "recommendations": []}}
    
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
    
    # Если Steam вернул ответ
    if "openid.identity" in params:
        # Извлекаем SteamID из ссылки
        sid = params["openid.identity"].split("/")[-1]
        
        # 1. Задаем значения по умолчанию (на случай сбоя API)
        user_name = "Steam User"
        user_avatar = "https://avatars.akamai.steamstatic.com/fef49e7fa7e1997310d705b2a6158ff8dc1cdfeb_full.jpg"
        
        # 2. Формируем URL для запроса профиля (ВАЖНО: api_url создается здесь)
        api_url = f"https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/?key={STEAM_API_KEY}&steamids={sid}"
        
        async with httpx.AsyncClient() as client:
            try:
                # 3. Делаем запрос к Steam за именем и аватаркой
                resp = await client.get(api_url, timeout=10.0)
                
                if resp.status_code == 200:
                    data = resp.json()
                    players = data.get('response', {}).get('players')
                    if players:
                        player = players[0]
                        # Сохраняем реальное имя
                        user_name = player.get('personaname', 'Steam User')
                        # Сохраняем самую четкую аватарку
                        user_avatar = player.get('avatarfull') or player.get('avatarmedium') or user_avatar
                else:
                    print(f"❌ Steam API Error: {resp.status_code}")
            except Exception as e:
                # Теперь эта ошибка не вылетит из-за api_url, так как она определена выше
                print(f"❌ Auth Error: {e}")

        # 4. Сохраняем всё в куки
        resp = RedirectResponse("/")
        resp.set_cookie("user_steam_id", sid, max_age=2592000)
        resp.set_cookie("user_name", quote(user_name), max_age=2592000)
        resp.set_cookie("user_avatar", user_avatar, max_age=2592000)
        return resp
    
    # Если данных нет, просто возвращаем на главную
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
    # Добавляем "or ''", чтобы вместо None всегда была пустая строка
    uname = unquote(request.cookies.get("user_name") or "")
    uavatar = request.cookies.get("user_avatar") or "" 
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "user_id": uid, 
        "user_name": uname, 
        "user_avatar": uavatar
    })

## if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, proxy_headers=True, forwarded_allow_ips="*")