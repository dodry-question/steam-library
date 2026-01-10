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

from groq import Groq
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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
print(f"DEBUG: Groq Key loaded: {'Yes' if GROQ_API_KEY else 'No'}")

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

async def fetch_steam_store_data(client: httpx.AsyncClient, app_id: int):
    async with STORE_API_LOCK:
        # Пытаемся получить RU регион
        data = await request_store(client, app_id, region="ru")
        
        if data == "RETRY_LATER":
            return "RETRY_LATER", False
            
        sid_str = str(app_id)
        is_fallback = False
        
        # Если в RU не удалось (регионлок), пробуем US
        if not data or not data.get(sid_str, {}).get('success'):
            await asyncio.sleep(0.5) # Маленькая пауза между попытками регионов
            data = await request_store(client, app_id, region="us")
            is_fallback = True
            
        return (data if data != "RETRY_LATER" else None), is_fallback
    
# Вспомогательная функция (обновите ее, чтобы принимала строку)
async def request_store(client, app_id, region="ru"):
    url = "https://store.steampowered.com/api/appdetails"
    params = {"appids": str(app_id), "cc": region, "l": "russian"}
    headers = {"User-Agent": "Mozilla/5.0"} 

    try:
        resp = await client.get(url, params=params, headers=headers, timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            return "RETRY_LATER" # Специальный маркер для блокировки
    except: 
        return None
    
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
    ids = payload.steam_ids
    playtimes = payload.playtimes
    names_map = payload.game_names
    cutoff = datetime.now() - timedelta(hours=24)

    # 1. Сначала отдаем ВСЁ, что есть в БД (это вылетит мгновенно)
    with Session(engine) as session:
        existing_games = session.exec(select(Game).where(Game.steam_id.in_(ids))).all()
        existing_map = {g.steam_id: g for g in existing_games}
        
        needed_from_steam = []
        for sid in ids:
            game = existing_map.get(sid)
            if game and game.last_updated > cutoff:
                d = game.model_dump()
                if d.get('last_updated'): d['last_updated'] = d['last_updated'].isoformat()
                d['playtime_forever'] = playtimes.get(sid, 0)
                yield json.dumps(d, ensure_ascii=False) + "\n"
            else:
                needed_from_steam.append(sid)

    # 2. Если чего-то нет в базе, идем в Store API по одному
    if needed_from_steam:
        async with httpx.AsyncClient() as client:
            for sid in needed_from_steam:
                store_resp, is_fallback = await fetch_steam_store_data(client, sid)
                store_resp, is_fallback = await fetch_steam_store_data(client, sid)

                if store_resp == "RETRY_LATER" or store_resp is None:
                    res = {
                        "steam_id": sid,
                        "name": names_map.get(sid, ""),
                        "price_str": "—", 
                        "genres": "",
                        "discount_percent": 0
                    }
                    yield json.dumps(res, ensure_ascii=False) + "\n"
                    continue 

                sid_str = str(sid)
                raw_data = store_resp.get(sid_str, {}) if store_resp else {}
                game_obj = parse_game_obj(sid, raw_data, names_map.get(sid, ""), is_fallback)

                with Session(engine) as session:
                    # Сохраняем/Обновляем
                    existing = session.exec(select(Game).where(Game.steam_id == sid)).first()
                    if existing:
                        if raw_data.get('success'):
                            for k, v in game_obj.model_dump(exclude={"id", "steam_id"}).items():
                                setattr(existing, k, v)
                        existing.last_updated = datetime.now()
                        session.add(existing)
                        res = existing.model_dump()
                    else:
                        session.add(game_obj)
                        res = game_obj.model_dump()
                    session.commit()
                    
                    if res.get('last_updated'): res['last_updated'] = res['last_updated'].isoformat()
                    res['playtime_forever'] = playtimes.get(sid, 0)
                    yield json.dumps(res, ensure_ascii=False) + "\n"
                
                await asyncio.sleep(0.8) # Пауза вежливости для Steam

@app.get("/api/get-games-list")
async def get_games_list(request: Request, user_id: Optional[str] = None):
    # (код получения target_id оставляем)
    target_id = await resolve_steam_id(user_id) if user_id else request.cookies.get("user_steam_id")
    if not target_id: return {"error": "ID не найден"}

    # Запрашиваем только базовый список (это быстро даже для 3000 игр)
    url = f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={STEAM_API_KEY}&steamid={target_id}&format=json&include_appinfo=1&include_played_free_games=1"
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, timeout=20.0)
            data = resp.json()
            if "response" in data and "games" in data["response"]:
                raw_games = data["response"]["games"]
                # Сортируем по времени игры сразу
                raw_games.sort(key=lambda x: x.get('playtime_forever', 0), reverse=True)
                
                games = []
                for g in raw_games:
                    appid = g["appid"]
                    games.append({
                        "appid": appid,
                        "name": g.get("name", f"App {appid}"),
                        "playtime_forever": g.get("playtime_forever", 0)
                    })
                return {"games": games}
            return {"error": "Профиль скрыт"}
        except Exception as e:
            return {"error": str(e)}
        
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
        mood = body.get("mood", "hidden gems")
        
        # Подготовка данных
        top_played = sorted(all_games, key=lambda x: x.get('playtime_forever', 0), reverse=True)[:10]
        core_names = ", ".join([g['name'] for g in top_played])
        
        sample_size = min(len(all_games), 80)
        owned_sample = random.sample(all_games, sample_size)
        owned_names = ", ".join([g['name'] for g in owned_sample])

        prompt = (
            f"Ты игровой эксперт. Игрок любит: {core_names}.\n"
            f"Найди 3 игры в Steam для настроения '{mood}'.\n"
            f"ПРАВИЛА:\n"
            f"- НЕ предлагай: {owned_names}.\n"
            f"- ФОРМАТ: Name: <название> | Based on: <игра из списка выше> | Reason: <почему подходит>\n"
            f"Отвечай ТОЛЬКО этими 3 строками на русском."
        )

        # СПИСОК МОДЕЛЕЙ (Если первая занята, пробуем следующую)
        # Все они бесплатные (:free) и быстрые
        MODELS_TO_TRY = [
            "google/gemini-2.0-flash-exp:free",            # Самая быстрая (Google)
            "meta-llama/llama-3.2-11b-vision-instruct:free", # Новая Llama (Meta)
            "microsoft/phi-3-medium-128k-instruct:free",   # Очень быстрая (Microsoft)
            "huggingfaceh4/zephyr-7b-beta:free"            # Стабильная резервная
        ]

        async with httpx.AsyncClient() as client:
            last_error = ""
            
            # Цикл перебора моделей
            for model_name in MODELS_TO_TRY:
                try:
                    print(f"🔄 Пробуем модель: {model_name}...")
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                            "HTTP-Referer": "http://localhost:8001",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}]
                        },
                        timeout=25.0
                    )
                    
                    result = resp.json()
                    
                    # Если ошибка API (например 429 или 404), идем к следующей модели
                    if "error" in result:
                        err_msg = result['error'].get('message', 'Unknown error')
                        print(f"⚠️ Ошибка модели {model_name}: {err_msg}")
                        last_error = err_msg
                        continue 
                    
                    # Если всё ок — берем текст и выходим из цикла
                    if "choices" in result and len(result["choices"]) > 0:
                        text = result['choices'][0]['message']['content']
                        print(f"✅ Успех ({model_name})!\n--- AI RESPONSE ---\n{text}")
                        
                        # Парсинг ответа
                        recs = []
                        for line in text.split('\n'):
                            line = line.strip()
                            if "|" in line:
                                try:
                                    parts = line.split("|")
                                    if len(parts) >= 3:
                                        g_name = re.sub(r'^(Name:|Название:|[\d\.\s]+)', '', parts[0], flags=re.I).strip()
                                        based_on = re.sub(r'^(Based on:|Основано на:)', '', parts[1], flags=re.I).strip()
                                        reason = re.sub(r'^(Reason:|Причина:)', '', parts[2], flags=re.I).strip()

                                        real_id = await search_steam_game(client, g_name)
                                        if real_id:
                                            recs.append({
                                                "steam_id": real_id,
                                                "name": g_name,
                                                "based_on": based_on,
                                                "ai_reason": reason,
                                                "image_url": f"https://cdn.akamai.steamstatic.com/steam/apps/{real_id}/header.jpg"
                                            })
                                except: continue
                        
                        return {"content": {"recommendations": recs}}
                        
                except Exception as e:
                    print(f"⚠️ Сбой соединения с {model_name}: {e}")
                    continue

            # Если ни одна модель не сработала
            print("❌ Все модели недоступны.")
            return {"content": {"error": f"Все серверы ИИ заняты. Последняя ошибка: {last_error}"}}

    except Exception as e:
        print(f"❌ Critical AI Error: {e}")
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