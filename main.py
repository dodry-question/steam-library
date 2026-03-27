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
    # Очищаем имя от лишних символов для поиска (оставляем только буквы и цифры)
    clean_name = re.sub(r'[^\w\s]', '', name).lower()
    
    params = {"term": name, "l": "russian", "cc": "ru"}
    try:
        resp = await client.get(search_url, params=params, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("total") > 0:
                items = data["items"]
                
                # 1. Попытка найти ТОЧНОЕ совпадение по названию
                for item in items:
                    item_name_clean = re.sub(r'[^\w\s]', '', item["name"]).lower()
                    if item_name_clean == clean_name:
                        return item["id"]
                
                # 2. Если точного нет, берем первый результат
                return items[0]["id"]
    except: pass
    return None

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

async def fetch_steam_store_data(client: httpx.AsyncClient, app_id: int):
    async with STORE_API_LOCK:
        # Пытаемся получить RU регион
        data = await request_store(client, app_id, region="ru")
        
        if data == "RETRY_LATER":
            return "RETRY_LATER", False
            
        sid_str = str(app_id)
        is_fallback = False
        
        # Если в RU не удалось (регионлок или просто нет данных), пробуем US
        if not data or not data.get(sid_str, {}).get('success'):
            await asyncio.sleep(0.5) # Маленькая пауза между попытками регионов
            data = await request_store(client, app_id, region="us")
            is_fallback = True
            
        return (data if data != "RETRY_LATER" else None), is_fallback

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

@app.post("/api/games-batch")
async def get_games_batch(payload: BatchRequest):
    return StreamingResponse(game_generator(payload), media_type="application/x-ndjson")

# --- ГЕНЕРАТОР (Который у вас отсутствовал) ---

async def game_generator(payload: BatchRequest):
    ids = payload.steam_ids
    playtimes = payload.playtimes
    names_map = payload.game_names
    # Считаем данные устаревшими через 24 часа
    cutoff = datetime.now() - timedelta(hours=24)

    # 1. Сначала отдаем ВСЁ, что есть в БД (это происходит мгновенно)
    with Session(engine) as session:
        existing_games = session.exec(select(Game).where(Game.steam_id.in_(ids))).all()
        existing_map = {g.steam_id: g for g in existing_games}
        
        needed_from_steam = []
        for sid in ids:
            game = existing_map.get(sid)
            # Если игра есть в базе и обновлялась недавно — отдаем сразу
            if game and game.last_updated > cutoff:
                d = game.model_dump()
                if d.get('last_updated'): d['last_updated'] = d['last_updated'].isoformat()
                d['playtime_forever'] = playtimes.get(sid, 0)
                yield json.dumps(d, ensure_ascii=False) + "\n"
            else:
                needed_from_steam.append(sid)

    # 2. Если чего-то нет в базе, идем в Store API по одной игре
    if needed_from_steam:
        async with httpx.AsyncClient() as client:
            for sid in needed_from_steam:
                store_resp, is_fallback = await fetch_steam_store_data(client, sid)
                
                # Если Steam заблокировал запросы (429)
                if store_resp == "RETRY_LATER":
                    # Делаем паузу и отдаем "пустышку", чтобы фронтенд не висел вечно
                    # (пользователь увидит игру, но без цены)
                    await asyncio.sleep(20) 
                    yield json.dumps({
                        "steam_id": sid,
                        "name": names_map.get(sid, ""),
                        "price_str": "—",
                        "genres": "",
                        "discount_percent": 0
                    }, ensure_ascii=False) + "\n"
                    continue

                sid_str = str(sid)
                raw_data = store_resp.get(sid_str, {}) if store_resp else {}
                
                # Парсим полученные данные
                game_obj = parse_game_obj(sid, raw_data, names_map.get(sid, ""), is_fallback)

                # Сохраняем в базу
                with Session(engine) as session:
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
                
                # Пауза вежливости для Steam (чтобы не словить бан)
                await asyncio.sleep(0.8)

# --- API ---

@app.post("/api/recommend")
async def recommend(request: Request):
    try:
        body = await request.json()
        all_games = body.get("games", [])
        mood = body.get("mood", "hidden gems")
        
        # НОВОЕ: Достаем историю рекомендаций
        already_recommended = body.get("already_recommended", [])
        
        # 1. Подготовка данных: любимые игры (для основы)
        top_played = sorted(all_games, key=lambda x: x.get('playtime_forever', 0), reverse=True)[:15]
        core_names = ", ".join([g['name'] for g in top_played])
        
        # 2. Исключения: игры, которые уже есть (берем топ 300, чтобы не советовал их)
        exclusions = sorted(all_games, key=lambda x: x.get('playtime_forever', 0), reverse=True)[:300]
        owned_names = ", ".join([g['name'] for g in exclusions])

        # НОВОЕ: Формируем дополнительное правило, если история не пустая
        history_rule = f"\n2. ТАКЖЕ ЗАПРЕЩЕНО советовать эти игры (ты их уже рекомендовал ранее): {', '.join(already_recommended)}." if already_recommended else ""

        # 3. Промпт с жестким требованием вернуть JSON
        prompt = f"""
Ты игровой эксперт. Игрок любит эти игры: {core_names}.
Посоветуй ровно 3 игры в Steam, которые подойдут под настроение: '{mood}'.
СТРОГИЕ ПРАВИЛА:
1. ЗАПРЕЩЕНО советовать игры, которые уже есть у игрока: {owned_names}.{history_rule}
3. Твой ответ должен быть СТРОГО в формате валидного JSON-массива, без Markdown разметки, без лишних слов.
Формат ответа:
[
  {{"name": "Название игры", "based_on": "Название игры из списка игрока, на которую она похожа", "reason": "Краткая причина на русском языке, почему она понравится"}}
]
"""
        # 4. Настройки подключения к VseGPT
        API_BASE_URL = os.environ.get("VSEGPT_BASE_URL", "https://api.vsegpt.ru/v1/chat/completions")
        API_KEY = os.environ.get("VSEGPT_API_KEY")

        if not API_KEY:
            print("❌ ОШИБКА: Ключ VSEGPT_API_KEY не найден в .env")
            return {"content": {"error": "API ключ ИИ не настроен."}}

        async with httpx.AsyncClient() as client:
            print(f"🔄 Отправляем запрос к VseGPT (gpt-4o-mini)...")
            
            # Запрос к нейросети
            resp = await client.post(
                API_BASE_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini", # Самая оптимальная по цене/качеству модель
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                },
                timeout=40.0
            )
            
            result = resp.json()
            
            # Обработка ошибок от самого сервиса VseGPT
            if "error" in result:
                err_msg = result['error'].get('message', 'Неизвестная ошибка')
                print(f"⚠️ Ошибка API VseGPT: {err_msg}")
                return {"content": {"error": "Ошибка на сервере ИИ. Проверьте консоль."}}
            
            # Успешный ответ
            if "choices" in result and len(result["choices"]) > 0:
                raw_text = result['choices'][0]['message']['content'].strip()
                print(f"✅ ИИ ответил:\n{raw_text}") # Выводим в терминал для проверки
                
                # Очищаем текст от Markdown тегов (ИИ любит оборачивать JSON в ```json ... ```)
                clean_text = re.sub(r"^```json", "", raw_text, flags=re.MULTILINE)
                clean_text = re.sub(r"^```", "", clean_text, flags=re.MULTILINE).strip()
                
                try:
                    # Превращаем текст в настоящий массив Python
                    ai_recommendations = json.loads(clean_text)
                    recs = []
                    
                    for item in ai_recommendations:
                        g_name = item.get("name", "")
                        based_on = item.get("based_on", "")
                        reason = item.get("reason", "")
                        
                        # Ищем реальный ID игры в Steam, чтобы сделать кликабельную карточку
                        real_id = await search_steam_game(client, g_name)
                        if real_id:
                            recs.append({
                                "steam_id": real_id,
                                "name": g_name,
                                "based_on": based_on,
                                "ai_reason": reason,
                                "image_url": f"https://cdn.akamai.steamstatic.com/steam/apps/{real_id}/header.jpg"
                            })
                            
                    if len(recs) > 0:
                        return {"content": {"recommendations": recs}}
                    else:
                        return {"content": {"error": "ИИ посоветовал игры, но Steam не смог их найти."}}
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Ошибка парсинга JSON: {e}\nТекст был: {clean_text}")
                    return {"content": {"error": "ИИ выдал ответ в неверном формате."}}
                    
            return {"content": {"error": "Пустой ответ от ИИ."}}

    except Exception as e:
        print(f"❌ Критическая ошибка ИИ: {e}")
        return {"content": {"error": str(e)}}

@app.post("/api/recommend-selected")
async def recommend_selected(request: Request):
    try:
        body = await request.json()
        all_games = body.get("games", [])
        target_games = body.get("target_games", [])
        mood = body.get("mood", "hidden gems")
        
        # НОВОЕ: Достаем историю рекомендаций
        already_recommended = body.get("already_recommended", [])
        
        if not target_games:
            return {"content": {"error": "Игры не выбраны"}}
            
        targets_str = ", ".join(target_games)
        
        exclusions = sorted(all_games, key=lambda x: x.get('playtime_forever', 0), reverse=True)[:350]
        owned_names = ", ".join([g['name'] for g in exclusions])

        # НОВОЕ: Правило для истории
        history_rule = f"\n2. ТАКЖЕ ЗАПРЕЩЕНО советовать эти игры (ты их уже рекомендовал ранее): {', '.join(already_recommended)}." if already_recommended else ""

        prompt = f"""
Ты игровой эксперт. Игрок выбрал эти игры из своей библиотеки: {targets_str}.
Посоветуй ровно 3 игры в Steam, которые максимально похожи на этот набор игр (по геймплею, атмосфере, жанру), учитывая настроение: '{mood}'.
СТРОГИЕ ПРАВИЛА:
1. ЗАПРЕЩЕНО советовать игры, которые уже есть у игрока: {owned_names}.{history_rule}
3. Ответ СТРОГО в формате JSON-массива, без Markdown, без лишних слов.
Формат ответа:
[
  {{"name": "Название игры", "based_on": "На какую из выбранных игр она больше всего похожа", "reason": "Кратко на русском, почему она подойдет этому игроку (смешанная механика, похожий сеттинг и т.д.)"}}
]
"""

        API_BASE_URL = os.environ.get("VSEGPT_BASE_URL", "https://api.vsegpt.ru/v1/chat/completions")
        API_KEY = os.environ.get("VSEGPT_API_KEY")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                API_BASE_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6 # Чуть пониже температура для большей точности
                },
                timeout=40.0
            )
            
            result = resp.json()
            if "choices" in result and len(result["choices"]) > 0:
                raw_text = result['choices'][0]['message']['content'].strip()
                clean_text = re.sub(r"^```json", "", raw_text, flags=re.MULTILINE)
                clean_text = re.sub(r"^```", "", clean_text, flags=re.MULTILINE).strip()
                clean_text = re.sub(r',\s*([\]}])', r'\1', clean_text) # Удаляет запятые перед закрывающими скобками
                
                try:
                    ai_recommendations = json.loads(clean_text)
                    recs = []
                    for item in ai_recommendations:
                        g_name = item.get("name", "")
                        based_on = item.get("based_on", "")
                        reason = item.get("reason", "")
                        real_id = await search_steam_game(client, g_name)
                        if real_id:
                            recs.append({
                                "steam_id": real_id,
                                "name": g_name,
                                "based_on": based_on,
                                "ai_reason": reason,
                                "image_url": f"https://cdn.akamai.steamstatic.com/steam/apps/{real_id}/header.jpg"
                            })
                    if recs: return {"content": {"recommendations": recs}}
                except Exception as e:
                    print(f"Ошибка парсинга: {e}")
                    
            return {"content": {"error": "Не удалось получить список похожих игр."}}

    except Exception as e:
        return {"content": {"error": str(e)}}

@app.get("/api/get-games-list")
async def get_games_list(request: Request, user_id: Optional[str] = None):
    # (код получения target_id оставляем)
    target_id = await resolve_steam_id(user_id) if user_id else request.cookies.get("user_steam_id")
    if not target_id: return {"error": "ID не найден"}

    # Запрашиваем только базовый список
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
    input_str = input_str.strip().strip('/')
    
    # 1. Ищем 17-значный SteamID64 (он всегда начинается с 7656119...) в любом месте строки
    match_64 = re.search(r'\b(7656119[0-9]{10})\b', input_str)
    if match_64:
        return match_64.group(1)
        
    # 2. Если это ссылка с кастомным URL (vanity name)
    if 'steamcommunity.com/id/' in input_str:
        clean = input_str.split('steamcommunity.com/id/')[-1].split('/')[0]
    else:
        # Если ввели просто текст или кусок ссылки
        clean = input_str.split('/')[-1]
        
    # 3. Делаем запрос к Steam для расшифровки кастомного имени
    url = f"https://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/?key={STEAM_API_KEY}&vanityurl={clean}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url)
            d = resp.json()
            if d.get('response', {}).get('success') == 1:
                return d['response']['steamid']
        except: pass
        
    return None

@app.post("/api/add-game")
async def add_game_manual(steam_id: int = Form(...)):
    payload = BatchRequest(steam_ids=[steam_id], playtimes={steam_id: 0}, game_names={})
    async for item in game_generator(payload):
        return json.loads(item)
    return {"error": "Не удалось загрузить"}

# --- ИИ ---

    
@app.get("/login")
async def login(request: Request): # Обязательно добавляем (request: Request)
    # Код сам определяет, запущен он на localhost или в Amvera (https)
    scheme = request.headers.get("x-forwarded-proto", "http")
    host = request.headers.get("host")
    current_domain = f"{scheme}://{host}"
    
    params = {
        "openid.ns": "http://specs.openid.net/auth/2.0",
        "openid.mode": "checkid_setup",
        "openid.return_to": f"{current_domain}/auth", # Используем динамический домен
        "openid.realm": f"{current_domain}",          # Используем динамический домен
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
