# 🧪 Быстрый старт с тестами

## Что добавлено в проект:

```
D:\SteamProject/
├── tests/
│   ├── __init__.py
│   ├── test_sanitize.py      # 13 тестов защиты от Prompt Injection
│   ├── test_api.py            # 12 тестов API эндпоинтов
│   └── README.md              # Подробная документация
├── pytest.ini                 # Конфигурация pytest
└── requirements-dev.txt       # Зависимости для тестирования
```

## Команды для работы с тестами:

### 1. Установка (один раз):
```bash
pip install -r requirements-dev.txt
```

### 2. Запуск всех тестов:
```bash
pytest
```

### 3. Запуск с подробным выводом:
```bash
pytest -v
```

### 4. Запуск конкретного файла:
```bash
pytest tests/test_sanitize.py
pytest tests/test_api.py
```

### 5. Запуск одного теста:
```bash
pytest tests/test_sanitize.py::TestSanitizeCustomQuery::test_dangerous_keyword_ignore
```

## Что тестируется:

### ✅ Защита от Prompt Injection (13 тестов)
- Блокировка опасных слов: ignore, system, prompt, assistant
- Ограничение длины до 80 символов
- Case-insensitive проверка
- Безопасные игровые запросы проходят

### ✅ API эндпоинты (12 тестов)
- `/api/recommend` - рекомендации с запросом и без
- `/api/recommend-selected` - рекомендации по выбранным играм
- Rate limiting (10 запросов/минуту)
- Авторизация (logout, auth-url)

## Результат первого запуска:

**25 тестов, все прошли успешно! ✅**

```
======================== 25 passed in 101.80s =========================
```

## Когда запускать тесты:

- ✅ Перед каждым git push
- ✅ После изменения логики API
- ✅ После изменения функции sanitize_custom_query
- ✅ Перед деплоем на Amvera

## Следующий шаг: CI/CD

Тесты готовы для интеграции с GitHub Actions. Создай файл `.github/workflows/test.yml`:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pip install -r requirements-dev.txt
      - run: pytest -v
```

Теперь при каждом push тесты будут запускаться автоматически!
