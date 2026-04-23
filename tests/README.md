# Steam Library Manager - Тесты

## Установка зависимостей для тестирования

```bash
pip install -r requirements-dev.txt
```

## Запуск тестов

### Запустить все тесты:
```bash
pytest
```

### Запустить с подробным выводом:
```bash
pytest -v
```

### Запустить конкретный файл:
```bash
pytest tests/test_sanitize.py
pytest tests/test_api.py
```

### Запустить конкретный тест:
```bash
pytest tests/test_sanitize.py::TestSanitizeCustomQuery::test_dangerous_keyword_ignore
```

### Показать покрытие кода (coverage):
```bash
pip install pytest-cov
pytest --cov=main --cov-report=html
```

## Структура тестов

```
tests/
├── __init__.py              # Пустой файл для Python пакета
├── test_sanitize.py         # Тесты защиты от Prompt Injection
└── test_api.py              # Тесты API эндпоинтов
```

## Что тестируется

### test_sanitize.py (13 тестов)
- ✅ Обработка None и пустых строк
- ✅ Ограничение длины до 80 символов
- ✅ Блокировка опасных ключевых слов (ignore, system, prompt, etc.)
- ✅ Case-insensitive проверка
- ✅ Безопасные игровые запросы проходят без изменений

### test_api.py (11 тестов)
- ✅ `/api/recommend` - с запросом и без
- ✅ `/api/recommend-selected` - выбор конкретных игр
- ✅ Защита от Prompt Injection в API
- ✅ Rate limiting (10 запросов в минуту)
- ✅ `/api/get-games-list` - получение библиотеки
- ✅ Авторизация (logout, auth-url)

## Примеры вывода

### Успешный запуск:
```
tests/test_sanitize.py::TestSanitizeCustomQuery::test_none_input PASSED
tests/test_sanitize.py::TestSanitizeCustomQuery::test_dangerous_keyword_ignore PASSED
tests/test_api.py::TestRecommendAPI::test_recommend_with_custom_query PASSED

======================== 24 passed in 2.34s ========================
```

### Провалившийся тест:
```
tests/test_sanitize.py::TestSanitizeCustomQuery::test_max_length_80 FAILED

AssertionError: assert 100 == 80
```

## CI/CD интеграция

Тесты готовы для интеграции с GitHub Actions, GitLab CI или другими CI/CD системами.

Пример для GitHub Actions (`.github/workflows/test.yml`):
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install -r requirements-dev.txt
      - run: pytest -v
```
