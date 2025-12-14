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
    """
    Умная функция: пробует скачать пачкой. 
    Если Steam выдает ошибку 400 (Bad Request), пробует скачать игры по одной.
    """
    if not app_ids:
        return {}
    
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
        print(f"🌍 [STEAM API] Запрос {len(app_ids)} шт -> {region.upper()}...")
        resp = await client.get(url, params=params, headers=headers)
        
        if resp.status_code == 200:
            data = resp.json()
            if data is None: return {}
            return data
            
        elif resp.status_code == 429:
            print(f"!!! RATE LIMIT (429) {region} - Спим 5 сек !!!")
            await asyncio.sleep(5)
            return None 
            
        # Если ошибка 400 и мы запрашивали МНОГО игр — значит один из ID "битый".
        elif resp.status_code == 400 and len(app_ids) > 1:
            print(f"   ⚠️ Ошибка 400 (пачка не прошла). Разбиваем {len(app_ids)} игр поодиночке...")
            combined_data = {}
            
            for single_id in app_ids:
                await asyncio.sleep(0.2) 
                one_game_data = await fetch_steam_batch(client, [single_id], region)
                if one_game_data:
                    combined_data.update(one_game_data)
            
            return combined_data

        elif resp.status_code == 400 and len(app_ids) == 1:
            print(f"   ❌ ID {app_ids[0]} недопустим (400). Пропускаем.")
            return {}

        else:
            print(f"   ⚠️ Странный статус: {resp.status_code}")
            return {}
            
    except Exception as e:
        print(f"❌ Error fetching batch: {repr(e)}")
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
    """Генератор: игры по одной + ПОСЛЕДОВАТЕЛЬНЫЙ парсинг тегов"""
    requested_ids = payload.steam_ids
    playtimes = payload.playtimes
    
    ids_to_fetch = []
    
    cutoff_time = datetime.now() - timedelta(hours=24)

    # 1. Из базы
    with Session(engine) as session:
        existing_games = session.exec(select(Game).where(Game.steam_id.in_(requested_ids))).all()
        existing_map = {g.steam_id: g for g in existing_games}

        for sid in requested_ids:
            if sid in existing_map:
                game_obj = existing_map[sid]
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

    # 2. Грузим недостающие
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        def save_and_yield(sid, data_dict, region, tags=None):
            with Session(engine) as session:
                new_data = process_game_data(sid, data_dict, region, tags)
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
                return d

        # --- ШАГ 1: RU ---
        missing_after_ru = []
        await asyncio.sleep(0.5) 
        
        ru_data = await fetch_steam_batch(client, ids_to_fetch, 'ru')
        
        if ru_data:
            success_ids = []
            for sid in ids_to_fetch:
                s_sid = str(sid)
                if s_sid in ru_data and ru_data[s_sid]['success']:
                    success_ids.append(sid)
                else:
                    missing_after_ru.append(sid)

            # Качаем теги ПО ОЧЕРЕДИ
            tags_map = {}
            if success_ids:
                print(f"🏷️ Качаем метки для {len(success_ids)} игр (по очереди)...")
                for sid in success_ids:
                    tags = await fetch_store_tags(client, sid)
                    tags_map[sid] = tags
            
            # Сохраняем
            for sid in success_ids:
                s_sid = str(sid)
                game_json = ru_data[s_sid]
                
                temp_check = process_game_data(sid, game_json, 'ru')
                if temp_check.price_str == "Не продается":
                    missing_after_ru.append(sid)
                else:
                    user_tags = tags_map.get(sid)
                    result = save_and_yield(sid, game_json, 'ru', user_tags)
                    yield json.dumps(result) + "\n"
        else:
            missing_after_ru = list(ids_to_fetch)

        # --- ШАГ 2: KZ ---
        missing_after_kz = []
        if missing_after_ru:
            await asyncio.sleep(1.0) 
            kz_data = await fetch_steam_batch(client, missing_after_ru, 'kz')
            if kz_data:
                for sid in missing_after_ru:
                    s_sid = str(sid)
                    if s_sid in kz_data and kz_data[s_sid]['success']:
                        result = save_and_yield(sid, kz_data[s_sid], 'kz', None)
                        yield json.dumps(result) + "\n"
                    else:
                        missing_after_kz.append(sid)
            else:
                missing_after_kz = missing_after_ru

        # --- ШАГ 3: US ---
        if missing_after_kz:
             await asyncio.sleep(1.0)
             us_data = await fetch_steam_batch(client, missing_after_kz, 'us')
             if us_data:
                for sid in missing_after_kz:
                    s_sid = str(sid)
                    if s_sid in us_data and us_data[s_sid]['success']:
                         result = save_and_yield(sid, us_data[s_sid], 'us', None)
                         yield json.dumps(result) + "\n"

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
    # 1. АНАЛИЗ ВКУСОВ
    # Берем только игры, где > 2 часов
    played_games = [g for g in user_games if g.get('playtime', 0) > 120]
    played_games.sort(key=lambda x: x.get('playtime', 0), reverse=True)
    
    # Берем Топ-5 для "Якоря"
    top_games = played_games[:5]
    
    if not top_games:
        top_games = user_games[:5] # Если мало играл, берем что есть

    games_str = ", ".join([f"{g.get('name')}" for g in top_games])

    # 2. НОВЫЙ ПРОМПТ (СТРАТЕГИЯ "ПОХОЖИЕ ТОВАРЫ")
    prompt = f"""
    Я геймер. Вот мои самые любимые игры (Топ-5): {games_str}.
    
    Твоя задача: Подобрать мне 3 игры, которые максимально ПОХОЖИ на эти.
    
    Правила:
    1. Ищи "игры-побратимы" (Game-Alikes). Если я люблю Skyrim, советуй крутые RPG (можно популярные, типа Ведьмака или Dragon Age).
    2. НЕ бойся советовать ХИТЫ. Мне нужны качественные игры, а не странное инди.
    3. Игнорируй жанры, которых НЕТ в моем топе (если я не играю в рогалики — не советуй их).
    4. Для каждой игры напиши причину: "Тот же геймплей, что в игре X, но с..."
    
    ФОРМАТ ОТВЕТА (СТРОГО СОБЛЮДАЙ):
    APPID: <ID игры в Steam> | NAME: <Название> | REASON: <Твой текст причины>
    APPID: <ID игры в Steam> | NAME: <Название> | REASON: <Твой текст причины>
    APPID: <ID игры в Steam> | NAME: <Название> | REASON: <Твой текст причины>
    
    (Никакого лишнего текста, только 3 строки по шаблону)
    """

    print("🤖 Запрос к ИИ (PollinationsAI)...")
    
    rec_results = []
    
    try:
        client = AsyncClient(provider=PollinationsAI)
        response = await client.chat.completions.create(
            model="openai",
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = response.choices[0].message.content
        
        # 3. ПАРСИНГ ОТВЕТА (ВЫТАСКИВАЕМ ID и ПРИЧИНУ)
        lines = ai_text.split('\n')
        ids_to_fetch = []
        reasons_map = {} # ID -> Причина
        
        for line in lines:
            if "APPID:" in line:
                try:
                    # Разбираем строку по разделителю "|"
                    parts = line.split("|")
                    
                    # Вытаскиваем ID (чистим от пробелов и букв)
                    raw_id = parts[0].replace("APPID:", "").strip()
                    app_id = int(re.search(r'\d+', raw_id).group()) # Ищем только цифры
                    
                    # Вытаскиваем причину
                    reason = parts[2].replace("REASON:", "").strip()
                    
                    ids_to_fetch.append(app_id)
                    reasons_map[app_id] = reason
                except Exception as e:
                    print(f"Ошибка парсинга строки ИИ: {line} -> {e}")

        if not ids_to_fetch:
            return {"error": "ИИ вернул некорректный формат. Попробуйте еще раз."}

        # 4. ПОЛУЧЕНИЕ КРАСИВЫХ КАРТИНОК И ЦЕН (Используем твою же функцию)
        print(f"📥 Качаем данные Steam для рекомендованных: {ids_to_fetch}")
        async with httpx.AsyncClient() as steam_client:
            # Используем fetch_steam_batch (он уже есть в твоем коде)
            steam_data = await fetch_steam_batch(steam_client, ids_to_fetch, 'ru')
            
            final_cards = []
            
            for app_id in ids_to_fetch:
                s_id = str(app_id)
                if steam_data and s_id in steam_data and steam_data[s_id]['success']:
                    # Обрабатываем через твою функцию process_game_data
                    game_obj = process_game_data(app_id, steam_data[s_id], 'ru')
                    
                    # Превращаем в словарь и добавляем текст от ИИ
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