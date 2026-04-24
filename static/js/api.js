// api.js - Все запросы к серверу

let syncController = null;

/**
 * Загружает библиотеку игр (свою или друга)
 */
async function loadLibrary(isFriend = false) {
    if (window.isProcessing) return;

    let targetIdentifier = 'me';
    if (isFriend) {
        const inp = document.getElementById('friend-id').value.trim();
        if (!inp) return;
        targetIdentifier = inp;
    }

    if (window.currentLoadedTarget === targetIdentifier && window.loadedGames.length > 0) {
        return;
    }

    if (syncController) {
        syncController.abort();
        syncController = null;
    }

    window.isProcessing = true;
    window.currentLoadedTarget = targetIdentifier;
    window.recommendedHistory = {};

    const statusBar = document.getElementById('status-bar');

    document.getElementById('empty-state').style.display = 'none';
    document.getElementById('container').innerHTML = '';
    window.loadedGames = [];
    document.getElementById('copy-btn').style.display = 'none';
    document.getElementById('ai-block').style.display = 'none';

    statusBar.innerHTML = '<span class="spinner"></span> Запрашиваем список у Steam...';

    try {
        let url = '/api/get-games-list';
        if (isFriend && targetIdentifier !== 'me') {
            url += `?user_id=${encodeURIComponent(targetIdentifier)}`;
        }

        const resp = await fetch(url);
        const data = await resp.json();

        if (data.games) {
            const total = data.games.length;
            window.loadedGames = data.games.map(g => ({
                steam_id: g.appid,
                name: g.name,
                playtime_forever: g.playtime_forever,
                image_url: `https://cdn.akamai.steamstatic.com/steam/apps/${g.appid}/header.jpg`,
                price_str: "...",
                genres: ""
            }));

            document.getElementById('copy-btn').style.display = 'inline-block';
            document.querySelector('.search-container').style.display = 'flex';

            statusBar.innerText = `Найдено ${total} игр. Отрисовываем...`;
            window.loadedGames.forEach(g => window.addCard(g));

            if (total > 100) {
                statusBar.innerHTML = `Список готов. <span style="color:#a4d007">Цены и жанры подгружаются в фоне...</span>`;
            } else {
                statusBar.innerHTML = `Загружено ${total} игр.`;
            }

            startBackgroundSync(data.games);
        } else if (data.error) {
            statusBar.innerHTML = `<span style="color: #ff5c5c;">Ошибка: ${data.error}</span>`;
            window.currentLoadedTarget = null;
        }
    } catch (e) {
        statusBar.innerText = "Ошибка соединения";
        window.currentLoadedTarget = null;
    } finally {
        window.isProcessing = false;
    }
}

/**
 * Фоновая синхронизация цен и жанров
 */
async function startBackgroundSync(rawList) {
    const statusBar = document.getElementById('status-bar');
    const total = rawList.length;
    let processedCount = 0;

    const playtimes = {};
    const names = {};
    const ids = rawList.map(g => {
        playtimes[g.appid] = g.playtime_forever;
        names[g.appid] = g.name;
        return g.appid;
    });

    syncController = new AbortController();

    try {
        const resp = await fetch('/api/games-batch', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ steam_ids: ids, playtimes: playtimes, game_names: names }),
            signal: syncController.signal
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();

            if (done) {
                statusBar.innerHTML = `Загрузка завершена! Обработано деталей: ${processedCount} из ${total}`;
                setTimeout(() => {
                    if(statusBar.innerText.includes('Загрузка завершена'))
                        statusBar.innerText = '';
                }, 5000);
                break;
            }

            buffer += decoder.decode(value);
            let lines = buffer.split("\n");
            buffer = lines.pop();

            for (let line of lines) {
                if (!line.trim()) continue;
                try {
                    const fullData = JSON.parse(line);
                    processedCount++;

                    if (total > 100 && processedCount % 5 === 0) {
                        statusBar.innerHTML = `Синхронизация цен... ${processedCount} / ${total}`;
                    }

                    const idx = window.loadedGames.findIndex(g => g.steam_id === fullData.steam_id);
                    if (idx !== -1) window.loadedGames[idx] = fullData;

                    window.updateCardDetails(fullData);
                } catch(e) {
                    console.error("Ошибка парсинга строки:", e);
                }
            }
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            console.log('Фоновая загрузка остановлена пользователем.');
        } else {
            console.error("Ошибка фоновой синхронизации:", e);
            statusBar.innerHTML = `<span style="color: #ff6c6c;">Фоновая загрузка прервана</span>`;
        }
    }
}

/**
 * Получить рекомендации от AI
 */
async function getAI() {
    if (window.loadedGames.length === 0) {
        alert("Сначала загрузите игры!");
        return;
    }

    // Проверка: блокируем повторный запрос пока идет обработка
    if (window.aiProcessing) {
        return;
    }

    const block = document.getElementById('ai-block');
    block.style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Получаем значение из текстового поля
    const customQuery = document.getElementById('ai-search-query')?.value.trim() || '';

    // ОТЛАДКА: Выводим в консоль что отправляем
    console.log("Отправляем запрос ИИ:", customQuery);

    // Используем запрос как ключ для истории (или 'default' если пусто)
    const historyKey = customQuery || 'default';
    let historyForCurrentQuery = window.recommendedHistory[historyKey] || [];

    let endpoint = '/api/recommend';
    let bodyData = {
        games: window.loadedGames,
        custom_query: customQuery || null,
        already_recommended: historyForCurrentQuery
    };
    let loadingText = 'ИИ анализирует ваш профиль (обычно занимает 10-15 сек)...';

    if (window.isSelectionMode && window.selectedGames.size > 0) {
        endpoint = '/api/recommend-selected';
        const targetGamesArray = Array.from(window.selectedGames);
        bodyData = {
            games: window.loadedGames,
            target_games: targetGamesArray,
            custom_query: customQuery || null,
            already_recommended: historyForCurrentQuery
        };
        loadingText = `ИИ подбирает игры, похожие на: <b>${targetGamesArray.join(', ')}</b>...`;
    }

    block.innerHTML = `<span class="spinner"></span> ${loadingText}`;
    window.aiProcessing = true; // Блокируем повторные запросы

    try {
        const sessionToken = localStorage.getItem('session_token');
        const headers = {'Content-Type': 'application/json'};
        if (sessionToken) {
            headers['X-Session-Token'] = sessionToken;
        }

        const resp = await fetch(endpoint, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(bodyData)
        });

        if (!resp.ok) {
            const errText = await resp.text();
            throw new Error(`Ошибка сервера ${resp.status}: ${errText.substring(0, 50)}`);
        }

        const data = await resp.json();

        if (!data || !data.content) {
            throw new Error(`Бэкенд вернул странный ответ: ${JSON.stringify(data).substring(0, 50)}`);
        }

        if (data.content.error) {
            block.innerHTML = `<span style="color: #ffcc00;">Сообщение от ИИ: ${data.content.error}</span>`;
            window.aiProcessing = false; // Разблокируем при ошибке
            return;
        }

        const recs = data.content.recommendations;

        if (recs && recs.length > 0) {
            if (!window.recommendedHistory[historyKey]) {
                window.recommendedHistory[historyKey] = [];
            }
            recs.forEach(r => {
                if (!window.recommendedHistory[historyKey].includes(r.name)) {
                    window.recommendedHistory[historyKey].push(r.name);
                }
            });

            let titleText = window.isSelectionMode ? "На основе вашего выбора:" : "Рекомендации профиля:";
            let html = `<h3 style="color:#fff; margin:0 0 10px 0;">${titleText}</h3><div class="ai-gallery">`;

            recs.forEach(r => {
                html += `
                <div class="card" style="animation:none; width: 300px;">
                    <a href="https://store.steampowered.com/app/${r.steam_id}" target="_blank" style="text-decoration:none; color:inherit; flex-grow:1;">
                        <img src="${r.image_url}" onerror="this.src='https://via.placeholder.com/280x130/171a21/c7d5e0?text=No+Image'">
                        <div class="info">
                            <h2>${r.name}</h2>
                            <div style="margin-bottom: 6px; display: flex; align-items: center;">
                                <span class="star-tooltip" title="Основано на: ${r.based_on}">★</span>
                                <span style="font-size: 11px; color: #697885; margin-left:5px;">Основано на: ${r.based_on}</span>
                            </div>
                            <div class="playtime" style="color: #c7d5e0; margin-top:2px; line-height:1.4; font-size: 12px;">
                                ${r.ai_reason}
                            </div>
                        </div>
                    </a>
                </div>`;
            });
            html += '</div>';
            block.innerHTML = html;

            window.aiProcessing = false; // Разблокируем после успешного ответа

            try {
                if (typeof window.clearSelection === "function") {
                    window.clearSelection();
                }
            } catch (cleanupError) {
                console.warn("Ошибка интерфейса при очистке (игнорируем):", cleanupError);
            }
        } else {
            block.innerHTML = 'ИИ не смог подобрать игры. Попробуйте сменить настроение.';
            window.aiProcessing = false; // Разблокируем если нет результатов
        }
    } catch(e) {
        console.error("Детальная ошибка во фронтенде:", e);
        block.innerHTML = `<span style="color: #ff5c5c;">Ошибка: <b>${e.message}</b></span><br><span style="font-size: 12px; color: gray;">Откройте консоль браузера (F12) для деталей.</span>`;
        window.aiProcessing = false; // Разблокируем при ошибке
    }
}

/**
 * Авторизация по ссылке
 */
async function loginWithUrl() {
    const input = document.getElementById('auth-url-input').value.trim();
    const errorDiv = document.getElementById('auth-error');
    const btn = document.getElementById('auth-url-btn');

    if (!input) {
        errorDiv.innerText = "Пожалуйста, введите ссылку";
        errorDiv.style.display = 'block';
        return;
    }

    btn.innerText = "Проверка...";
    btn.disabled = true;
    errorDiv.style.display = 'none';

    try {
        const response = await fetch('/auth-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: input })
        });

        const data = await response.json();

        if (data.success) {
            window.location.reload();
        } else {
            errorDiv.innerText = data.error || "Произошла ошибка";
            errorDiv.style.display = 'block';
        }
    } catch (e) {
        errorDiv.innerText = "Ошибка соединения с сервером";
        errorDiv.style.display = 'block';
    } finally {
        btn.innerText = "Привязать профиль";
        btn.disabled = false;
    }
}

// Экспортируем функции в глобальную область
window.loadLibrary = loadLibrary;
window.getAI = getAI;
window.loginWithUrl = loginWithUrl;
