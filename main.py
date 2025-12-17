from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Field, Session, SQLModel, create_engine, select
import httpx
import asyncio
import json
import re
import os
import random
import g4f
from g4f.client import AsyncClient
from fastapi.staticfiles import StaticFiles
from g4f.Provider import PollinationsAI
from urllib.parse import quote, unquote

# --- НАСТРОЙКИ ---
STEAM_API_KEY = os.environ.get("STEAM_API_KEY") 
MY_DOMAIN = os.environ.get("MY_DOMAIN")

# Курсы валют (примерные, обновятся при запуске)
RATE_KZT_TO_RUB = 0.21  
RATE_USD_TO_RUB = 95.0 

# --- База данных ---
class Game(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    steam_id: int = Field(index=True)
    name: str
    image_url: str
    genres: str | None = None
    price_str: str | None = None 
    discount_percent: int = 0
    last_updated: datetime = Field(default_factory=datetime.now)

# Модель для приема списка ID от фронтенда
class BatchRequest(SQLModel):
    steam_ids: List[int]
    playtimes: Dict[int, int]  # Словарь: id -> время в минутах

sqlite_file_name = "games.db"
engine = create_engine(f"sqlite:///{sqlite_file_name}")

async def update_currency_rates():
    """Обновляет курсы валют с сайта ЦБ РФ"""
    global RATE_KZT_TO_RUB, RATE_USD_TO_RUB
    
    url = "https://www.cbr-xml-daily.ru/daily_json.js"
    print("💱 Обновляем курсы валют...")
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            data = resp.json()
            
            usd_data = data["Valute"]["USD"]
            kzt_data = data["Valute"]["KZT"]
            
            RATE_USD_TO_RUB = usd_data["Value"] / usd_data["Nominal"]
            RATE_KZT_TO_RUB = kzt_data["Value"] / kzt_data["Nominal"]
            
            print(f"✅ Курсы ЦБ РФ загружены: USD={RATE_USD_TO_RUB:.2f}₽, KZT={RATE_KZT_TO_RUB:.4f}₽")
            
    except Exception as e:
        print(f"⚠️ Ошибка обновления курсов (используем стандартные): {e}")

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def on_startup():
    create_db_and_tables()
    await update_currency_rates()

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

async def fetch_steam_batch(client: httpx.AsyncClient, app_ids: List[int], region: str):
    if not app_ids:
        return {}
    
    # Базовый URL
    url = "https://store.steampowered.com/api/appdetails"
    params = {
        "appids": ",".join(map(str, app_ids)),
        "l": "russian",
        "cc": region
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        # Небольшая пауза, чтобы не дудосить, но параллельные запросы перекроют это ожидание
        await asyncio.sleep(random.uniform(0.5, 1.0))
        
        print(f"🌍 [STEAM API] Запрос {len(app_ids)} шт -> {region.upper()}...")
        resp = await client.get(url, params=params, headers=headers)
        
        if resp.status_code == 200:
            data = resp.json()
            return data if data else {}
            
        elif resp.status_code == 429:
            print(f"!!! RATE LIMIT (429) {region} - Спим 10 сек !!!")
            await asyncio.sleep(10)
            # Пробуем рекурсивно еще раз
            return await fetch_steam_batch(client, app_ids, region)
            
        # --- УМНОЕ РАЗБИЕНИЕ (BINARY SPLIT) ---
        elif resp.status_code == 400 and len(app_ids) > 1:
            print(f"   ⚠️ Ошибка 400. Делим пачку {len(app_ids)} на две части...")
            mid = len(app_ids) // 2
            group1 = app_ids[:mid]
            group2 = app_ids[mid:]
            
            # Запускаем обе половинки
            data1 = await fetch_steam_batch(client, group1, region)
            data2 = await fetch_steam_batch(client, group2, region)
            
            combined = {}
            if data1: combined.update(data1)
            if data2: combined.update(data2)
            return combined

        elif resp.status_code == 400 and len(app_ids) == 1:
            # Если даже одна игра выдает 400, значит её не существует
            return {}

    except Exception as e:
        print(f"❌ Ошибка соединения: {repr(e)}")
        return {}
    return {}

async def resolve_steam_id_from_url(input_str: str) -> Optional[str]:
    """
    Превращает ссылку, vanity url или ID в чистый SteamID64.
    Примеры входа:
    - https://steamcommunity.com/id/gabelogannewell
    - https://steamcommunity.com/profiles/76561197960287930
    - gabelogannewell
    - 76561197960287930
    """
    input_str = input_str.strip()
    
    # 1. Если это уже чистый ID (только цифры, длина 17)
    if input_str.isdigit() and len(input_str) == 17:
        return input_str

    # 2. Очищаем от URL части
    clean_str = input_str
    if "steamcommunity.com/profiles/" in input_str:
        clean_str = input_str.split("profiles/")[1].split("/")[0]
        if clean_str.isdigit(): return clean_str
    
    if "steamcommunity.com/id/" in input_str:
        clean_str = input_str.split("id/")[1].split("/")[0]

    # 3. Пытаемся разрешить Vanity URL через API
    url = f"http://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/?key={STEAM_API_KEY}&vanityurl={clean_str}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url)
            data = resp.json()
            if data['response']['success'] == 1:
                return data['response']['steamid']
        except:
            pass
            
    return None

async def fetch_store_tags(client: httpx.AsyncClient, app_id: int) -> str:
    """Парсим теги аккуратно, с задержкой, чтобы не получить бан"""
    url = f"https://store.steampowered.com/app/{app_id}/?l=russian"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Cookie": "birthtime=0; lastagecheckage=1-0-1900; wants_mature_content=1"
    }
    
    try:
        # Случайная пауза, имитация человека
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        resp = await client.get(url, headers=headers, timeout=5.0)
        if resp.status_code == 200:
            text = resp.text
            tags = re.findall(r'<a[^>]+class="app_tag"[^>]*>\s*([^<]+?)\s*</a>', text)
            if tags:
                clean_tags = [t.strip() for t in tags[:5] if t.strip() != '+']
                return ", ".join(clean_tags)
    except Exception as e:
        pass
    
    return None
    
def process_game_data(steam_id: int, data: dict, region: str, custom_tags: str = None) -> Game:
    game_data = data['data']
    name = game_data['name']
    image = game_data.get('header_image', '')
    
    # ЕСЛИ МЫ НАШЛИ НАРОДНЫЕ ТЕГИ - БЕРЕМ ИХ. ЕСЛИ НЕТ - ОФИЦИАЛЬНЫЕ ЖАНРЫ.
    if custom_tags:
        genres_str = custom_tags
    else:
        genres_list = [g['description'] for g in game_data.get('genres', [])]
        genres_str = ", ".join(genres_list)

    price_text = "Не продается"
    discount = 0

    if game_data.get('is_free'):
        price_text = "Бесплатно"
    elif 'price_overview' in game_data:
        price_obj = game_data['price_overview']
        discount = price_obj['discount_percent']
        
        if region == 'kz':
            price_val = price_obj['final'] / 100
            price_rub = int(price_val * RATE_KZT_TO_RUB)
            price_text = f"~{price_rub} ₽"
        elif region == 'us':
            price_val = price_obj['final'] / 100
            price_rub = int(price_val * RATE_USD_TO_RUB)
            price_text = f"~{price_rub} ₽"
        else:
            price_text = price_obj['final_formatted']

    return Game(
        steam_id=steam_id,
        name=name,
        image_url=image,
        genres=genres_str,
        price_str=price_text,
        discount_percent=discount
    )

# --- ЛОГИКА API ---

@app.post("/api/games-batch")
async def get_games_batch(payload: BatchRequest):
    return StreamingResponse(generate_games(payload), media_type="application/x-ndjson")

async def generate_games(payload: BatchRequest):
    requested_ids = payload.steam_ids
    playtimes = payload.playtimes
    
    # 1. Сначала отдаем то, что УЖЕ есть в базе (Мгновенно)
    ids_to_fetch = []
    cutoff_time = datetime.now() - timedelta(hours=24)

    with Session(engine) as session:
        existing_games = session.exec(select(Game).where(Game.steam_id.in_(requested_ids))).all()
        existing_map = {g.steam_id: g for g in existing_games}

        for sid in requested_ids:
            if sid in existing_map:
                game_obj = existing_map[sid]
                # Если данные свежие - отдаем сразу
                if game_obj.last_updated and game_obj.last_updated > cutoff_time:
                    d = game_obj.model_dump()
                    d['playtime_forever'] = playtimes.get(sid, 0)
                    d['last_updated'] = game_obj.last_updated.isoformat()
                    yield json.dumps(d) + "\n"
                else:
                    ids_to_fetch.append(sid)
            else:
                ids_to_fetch.append(sid)

    if not ids_to_fetch:
        return

    # 2. Настраиваем параллельность
    # SEMAPHORE = 3 (Безопасно) или 5 (Быстро, но риск бана выше)
    # Попробуй начать с 4
    sem = asyncio.Semaphore(4) 

    async with httpx.AsyncClient(timeout=45.0) as client:
        
        # Вспомогательная функция для обработки одной пачки (10-20 игр)
        async def process_chunk(chunk_ids):
            async with sem: # Ограничиваем кол-во одновременных потоков
                # Шаг 1: RU
                steam_data = await fetch_steam_batch(client, chunk_ids, 'ru')
                region = 'ru'
                
                # Если в RU пусто или не продается, пробуем KZ
                # (Упрощенная логика для скорости: проверяем только RU и KZ)
                missing_in_ru = []
                if not steam_data:
                    missing_in_ru = chunk_ids
                else:
                    for sid in chunk_ids:
                        s_sid = str(sid)
                        if s_sid not in steam_data or not steam_data[s_sid]['success']:
                            missing_in_ru.append(sid)
                        else:
                            # Проверяем "Не продается"
                            temp = process_game_data(sid, steam_data[s_sid], 'ru')
                            if temp.price_str == "Не продается":
                                missing_in_ru.append(sid)

                if missing_in_ru:
                    kz_data = await fetch_steam_batch(client, missing_in_ru, 'kz')
                    if kz_data:
                        if not steam_data: steam_data = {}
                        steam_data.update(kz_data)
                        region = 'kz' # Смешанный режим, но цену возьмем последнюю

                results = []
                
                # Шаг 2: Обработка и ТЕГИ
                # Чтобы ускорить, мы не будем делать отдельный цикл для тегов.
                # Мы возьмем ЖАНРЫ из JSON, которые уже скачались.
                # ЕСЛИ ты очень хочешь парсить HTML-теги, это замедлит всё в разы.
                # Ниже компромисс: если игр в пачке мало, грузим теги.
                
                for sid in chunk_ids:
                    s_sid = str(sid)
                    if s_sid in steam_data and steam_data[s_sid]['success']:
                        game_json = steam_data[s_sid]
                        
                        # --- ТЕГИ: Самое узкое место ---
                        # Чтобы было быстрее, пробуем брать genres из JSON.
                        # Раскомментируй строку ниже, если готов ждать ради тегов
                        custom_tags = await fetch_store_tags(client, sid) 
                        # custom_tags = None # Если хочешь скорость ракеты - оставь None
                        
                        with Session(engine) as session:
                            new_data = process_game_data(sid, game_json, region, custom_tags)
                            
                            # Сохранение в БД (Upsert)
                            existing = session.exec(select(Game).where(Game.steam_id == sid)).first()
                            if existing:
                                existing.name = new_data.name
                                existing.image_url = new_data.image_url
                                existing.genres = new_data.genres
                                existing.price_str = new_data.price_str
                                existing.discount_percent = new_data.discount_percent
                                existing.last_updated = datetime.now()
                                session.add(existing)
                                final_obj = existing
                            else:
                                session.add(new_data)
                                final_obj = new_data
                            
                            session.commit()
                            session.refresh(final_obj)
                            
                            d = final_obj.model_dump()
                            d['playtime_forever'] = playtimes.get(sid, 0)
                            d['last_updated'] = final_obj.last_updated.isoformat()
                            results.append(d)
                return results

        # 3. Разбиваем на пачки по 10 штук (было 10)
        # Уменьшим пачку до 6, чтобы реже ловить ошибку 400
        chunk_size = 6
        chunks = [ids_to_fetch[i:i + chunk_size] for i in range(0, len(ids_to_fetch), chunk_size)]
        
        # Создаем задачи
        tasks = [asyncio.create_task(process_chunk(chunk)) for chunk in chunks]
        
        # 4. Выдаем результат по мере готовности (as_completed)
        # Это значит, что игры будут появляться на экране СРАЗУ, как только скачалась любая пачка
        for completed_task in asyncio.as_completed(tasks):
            try:
                batch_results = await completed_task
                for game_res in batch_results:
                    yield json.dumps(game_res) + "\n"
            except Exception as e:
                print(f"Ошибка в батче: {e}")

# --- СТАНДАРТНЫЕ МАРШРУТЫ ---

@app.get("/login")
def login():
    steam_openid_url = "https://steamcommunity.com/openid/login"
    params = {
        "openid.ns": "http://specs.openid.net/auth/2.0",
        "openid.mode": "checkid_setup",
        "openid.return_to": f"{MY_DOMAIN}/auth",
        "openid.realm": f"{MY_DOMAIN}",
        "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
        "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
    }
    param_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return RedirectResponse(f"{steam_openid_url}?{param_string}")

@app.get("/auth")
async def auth(request: Request):
    params = request.query_params
    if "openid.identity" in params:
        steam_id64 = params["openid.identity"].split("/")[-1]
        api_url = f"http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/?key={STEAM_API_KEY}&steamids={steam_id64}"
        user_name = "Steam User"
        user_avatar = ""
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(api_url)
                data = resp.json()
                player = data['response']['players'][0]
                user_name = player['personaname']
                if 'avatarmedium' in player: user_avatar = player['avatarmedium']
                if 'avatarfull' in player: user_avatar = player['avatarfull']
            except: pass
        response = RedirectResponse(url="/")
        response.set_cookie(key="user_steam_id", value=steam_id64)
        response.set_cookie(key="user_name", value=quote(user_name))
        response.set_cookie(key="user_avatar", value=user_avatar)
        return response
    return RedirectResponse(url="/")

@app.get("/logout")
def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("user_steam_id")
    response.delete_cookie("user_name")
    response.delete_cookie("user_avatar")
    return response

@app.get("/api/get-games-list") # Переименовали для универсальности
@app.get("/api/get-games-list")
async def get_games_list(request: Request, user_id: Optional[str] = None):
    target_id = None
    
    # 1. Определяем ID
    if user_id:
        target_id = await resolve_steam_id_from_url(user_id)
    if not target_id:
        target_id = request.cookies.get("user_steam_id")
    
    if not target_id: 
        return {"error": "User not found", "games": []}

    print(f"📥 Загружаем библиотеку для ID: {target_id}")

    # 2. Получаем ИМЯ пользователя (Новый код)
    target_name = target_id # По умолчанию имя = ID
    async with httpx.AsyncClient() as client:
        try:
            # Запрашиваем инфо о профиле
            summary_url = f"http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/?key={STEAM_API_KEY}&steamids={target_id}"
            user_resp = await client.get(summary_url)
            user_data = user_resp.json()
            players = user_data.get('response', {}).get('players', [])
            if players:
                target_name = players[0].get('personaname', target_id)
        except Exception as e:
            print(f"⚠️ Не удалось получить имя профиля: {e}")

        # 3. Получаем список игр
        url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={STEAM_API_KEY}&steamid={target_id}&format=json&include_appinfo=1&include_played_free_games=1"
        try:
            resp = await client.get(url)
            data = resp.json()
            if "response" in data and "games" in data["response"]:
                games_list = [{"appid": g["appid"], "playtime": g.get("playtime_forever", 0)} for g in data["response"]["games"]]
                # Возвращаем и ID, и Имя
                return {
                    "target_id": target_id, 
                    "target_name": target_name, 
                    "games": games_list
                }
            else:
                 return {"error": "Profile private or empty", "games": []}
        except Exception as e:
            print(f"Error getting list: {e}")
            pass
            
    return {"games": []}

@app.post("/api/add-game")
async def add_game_api(steam_id: int = Form(...)):
    """Ручное добавление (исправлено под генератор)"""
    payload = BatchRequest(steam_ids=[steam_id], playtimes={steam_id: 0})
    generator = generate_games(payload)
    async for game_json_str in generator:
        return json.loads(game_json_str)
    return {"error": "Игра не найдена"}

async def get_ai_recommendations(user_games: list):
    # 1. АНАЛИЗ ВКУСОВ (Твой старый код подготовки данных)
    played_games = [g for g in user_games if g.get('playtime', 0) > 120]
    played_games.sort(key=lambda x: x.get('playtime', 0), reverse=True)
    top_games = played_games[:5]
    if not top_games:
        top_games = user_games[:5]
    
    games_str = ", ".join([f"{g.get('name')}" for g in top_games])

    # 2. ПРОМПТ (Чуть упростим для надежности)
    prompt = f"""
    Я люблю игры: {games_str}.
    Посоветуй 3 похожие игры.
    
    ФОРМАТ ОТВЕТА (СТРОГО):
    APPID: <ID> | NAME: <Название> | REASON: <Короткая причина>
    APPID: <ID> | NAME: <Название> | REASON: <Короткая причина>
    APPID: <ID> | NAME: <Название> | REASON: <Короткая причина>
    """

    print("🤖 Запрос к Pollinations AI (Direct)...")

    try:
        # --- ВОТ ЗДЕСЬ ИЗМЕНЕНИЯ: ПРЯМОЙ ЗАПРОС БЕЗ G4F ---
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Pollinations имеет простой API endpoint
            url = "https://text.pollinations.ai/"
            
            # Отправляем POST запрос
            response = await client.post(
                url, 
                json={
                    "messages": [
                        {"role": "system", "content": "Ты полезный помощник геймера."},
                        {"role": "user", "content": prompt}
                    ],
                    "model": "openai", # Или "mistral", "llama"
                    "seed": random.randint(1, 1000) # Случайность для разнообразия
                }
            )
            
            if response.status_code != 200:
                return {"error": f"Ошибка AI провайдера: {response.status_code}"}
                
            ai_text = response.text
        # ----------------------------------------------------

        # 3. ПАРСИНГ ОТВЕТА (Твой старый код)
        lines = ai_text.split('\n')
        ids_to_fetch = []
        reasons_map = {} 
        
        for line in lines:
            if "APPID:" in line:
                try:
                    parts = line.split("|")
                    raw_id = parts[0].replace("APPID:", "").strip()
                    app_id = int(re.search(r'\d+', raw_id).group())
                    reason = parts[2].replace("REASON:", "").strip()
                    ids_to_fetch.append(app_id)
                    reasons_map[app_id] = reason
                except Exception as e:
                    print(f"Ошибка парсинга: {e}")

        if not ids_to_fetch:
            # Если формат сломался, вернем сырой текст ошибки для отладки
            print(f"Сырой ответ AI: {ai_text}")
            return {"error": "ИИ ответил не по формату. Попробуй еще раз."}

        # 4. ЗАГРУЗКА ДАННЫХ STEAM (Твой старый код)
        print(f"📥 Качаем данные Steam: {ids_to_fetch}")
        async with httpx.AsyncClient() as steam_client:
            steam_data = await fetch_steam_batch(steam_client, ids_to_fetch, 'ru')
            final_cards = []
            for app_id in ids_to_fetch:
                s_id = str(app_id)
                if steam_data and s_id in steam_data and steam_data[s_id]['success']:
                    game_obj = process_game_data(app_id, steam_data[s_id], 'ru')
                    d = game_obj.model_dump()
                    d['ai_reason'] = reasons_map.get(app_id, "Рекомендация ИИ")
                    final_cards.append(d)
            
            return {"recommendations": final_cards}

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return {"error": f"Ошибка сервера: {e}"}
    
@app.post("/api/recommend")
async def recommend_endpoint(request: Request):
    try:
        data = await request.json()
        games = data.get("games", [])
        # Вызываем твою функцию
        recommendation = await get_ai_recommendations(games)
        return {"content": recommendation}
    except Exception as e:
        print(f"Ошибка API: {e}")
        return {"content": f"Ошибка сервера: {str(e)}"}

@app.get("/")
def home(request: Request):
    user_id = request.cookies.get("user_steam_id")
    user_name = request.cookies.get("user_name")
    user_avatar = request.cookies.get("user_avatar")
    if user_name: user_name = unquote(user_name)
    return templates.TemplateResponse("index.html", {
        "request": request, "user_id": user_id, "user_name": user_name, "user_avatar": user_avatar
    })

@app.get("/test")
def test_page():
    return "СЕРВЕР РАБОТАЕТ! ПРОБЛЕМА В ШАБЛОНАХ."