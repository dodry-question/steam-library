# Steam Library Manager

Веб-приложение для управления библиотекой Steam с AI-рекомендациями игр.

## Возможности

- 📊 Просмотр библиотеки Steam с актуальными ценами и скидками
- 🤖 AI-рекомендации игр на основе вашего профиля
- 🔍 Поиск и сортировка игр
- 📱 Адаптивный дизайн (desktop + mobile)
- 🎮 Просмотр библиотек друзей
- ⏱️ Статистика времени в играх

## Технологии

**Backend:**
- FastAPI (асинхронный веб-фреймворк)
- SQLModel (ORM для работы с БД)
- httpx (асинхронные HTTP запросы)
- SQLite (база данных)

**Frontend:**
- Vanilla JavaScript (модульная архитектура)
- CSS3 (адаптивный дизайн)
- Jinja2 (шаблонизатор)

**AI:**
- VseGPT API (GPT-4o-mini)
- Groq API

## Установка

### 1. Клонирование репозитория

```bash
git clone <your-repo-url>
cd SteamProject
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```env
STEAM_API_KEY=ваш_ключ_steam
GROQ_API_KEY=ваш_ключ_groq
VSEGPT_BASE_URL=https://api.vsegpt.ru/v1/chat/completions
VSEGPT_API_KEY=ваш_ключ_vsegpt
MY_DOMAIN=http://localhost:8000
```

**Где получить ключи:**
- Steam API Key: https://steamcommunity.com/dev/apikey
- Groq API Key: https://console.groq.com/
- VseGPT API Key: https://vsegpt.ru/

### 4. Запуск приложения

```bash
python main.py
```

Или через uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Приложение будет доступно по адресу: http://localhost:8000

## Структура проекта

```
SteamProject/
├── main.py                 # Основной файл приложения
├── games.db               # База данных SQLite
├── app.log                # Логи приложения
├── requirements.txt       # Зависимости Python
├── .env                   # Переменные окружения (не в Git!)
├── .gitignore            # Игнорируемые файлы
├── static/
│   ├── css/
│   │   ├── main.css      # Основные стили
│   │   └── mobile.css    # Мобильные стили
│   ├── js/
│   │   ├── app.js        # Главный файл приложения
│   │   ├── api.js        # API запросы
│   │   ├── ui.js         # Работа с интерфейсом
│   │   └── selection.js  # Логика выделения игр
│   └── favicon.png       # Иконка сайта
└── templates/
    └── index.html        # HTML шаблон
```

## Безопасность

### CORS
Настроен для конкретных доменов. Для локальной разработки разрешены:
- http://localhost:8000
- http://localhost:8001
- http://127.0.0.1:8000
- http://127.0.0.1:8001

Для продакшена добавьте свой домен в `main.py`:

```python
allow_origins=[
    "https://your-domain.com",
    "http://localhost:8000"
]
```

### Rate Limiting
AI запросы ограничены: **10 запросов в минуту на IP**.

Настройка в `main.py`:
```python
AI_RATE_LIMIT = 10  # запросов
AI_RATE_WINDOW = 60  # секунд
```

### Переменные окружения
**ВАЖНО:** Никогда не коммитьте `.env` файл в Git!

Проверьте `.gitignore`:
```
.env
*.db
app.log
```

## Деплой на Amvera

1. Создайте проект на https://amvera.ru
2. Подключите Git репозиторий
3. Добавьте переменные окружения в панели Amvera:
   - `STEAM_API_KEY`
   - `GROQ_API_KEY`
   - `VSEGPT_API_KEY`
   - `VSEGPT_BASE_URL`
   - `MY_DOMAIN` (ваш домен на Amvera)

4. Обновите CORS в `main.py`:
```python
allow_origins=[
    "https://your-app.amvera.io",
    "http://localhost:8000"
]
```

## Логирование

Логи сохраняются в файл `app.log` и выводятся в консоль.

Уровни логирования:
- `INFO` - общая информация
- `WARNING` - предупреждения
- `ERROR` - ошибки

Просмотр логов:
```bash
tail -f app.log
```

## Разработка

### Добавление новых функций

1. **Backend (Python):**
   - Добавьте эндпоинт в `main.py`
   - Используйте async/await для асинхронных операций
   - Добавьте логирование

2. **Frontend (JavaScript):**
   - API запросы → `static/js/api.js`
   - UI логика → `static/js/ui.js`
   - Стили → `static/css/main.css`

### Тестирование

Запустите приложение локально:
```bash
python main.py
```

Откройте в браузере: http://localhost:8000

## Известные проблемы

- Steam API может блокировать при частых запросах (используется rate limiting)
- Региональные ограничения игр (показывается цена из США)
- AI может не найти игру, если название неточное

## Лицензия

Этот проект создан в образовательных целях.

## Автор

Создано с помощью Gemini и Claude AI.

## Поддержка

Если возникли проблемы:
1. Проверьте логи в `app.log`
2. Убедитесь, что все ключи API корректны
3. Проверьте, что профиль Steam открыт (настройки приватности)
