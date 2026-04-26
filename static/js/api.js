// api.js - Все запросы к серверу

let syncController = null;

/**
 * Показать уведомление в мини-окне загрузки
 * @param {string} message - текст сообщения
 * @param {string} type - тип: 'info', 'success', 'warning', 'error', 'progress'
 * @param {number} percent - процент для прогресс-бара (только для type='progress')
 * @param {string} subtext - дополнительный текст снизу
 */
function showNotification(message, type = 'info', percent = 0, subtext = '') {
    const statusBar = document.getElementById('status-bar');

    let bgColor = 'rgba(23, 26, 33, 0.85)'; // default
    if (type === 'error') bgColor = 'rgba(139, 0, 0, 0.85)'; // темно-красный
    if (type === 'warning') bgColor = 'rgba(184, 134, 11, 0.85)'; // темно-желтый
    if (type === 'success') bgColor = 'rgba(0, 100, 0, 0.85)'; // темно-зеленый

    if (type === 'progress') {
        statusBar.innerHTML = `
            <div class="price-sync-container" style="background: ${bgColor};">
                <div class="price-sync-label">
                    <span>${message}</span>
                    <span class="price-sync-percent">${percent}%</span>
                </div>
                <div class="price-sync-bar">
                    <div class="price-sync-progress" style="width: ${percent}%"></div>
                </div>
                ${subtext ? `<div class="price-sync-count">${subtext}</div>` : ''}
            </div>
        `;
    } else {
        statusBar.innerHTML = `
            <div class="price-sync-container" style="background: ${bgColor};">
                <div class="price-sync-label">
                    <span>${message}</span>
                </div>
                ${subtext ? `<div class="price-sync-count">${subtext}</div>` : ''}
            </div>
        `;
    }
}

/**
 * Скрыть уведомление
 */
function hideNotification(delay = 5000) {
    const statusBar = document.getElementById('status-bar');
    setTimeout(() => {
        statusBar.innerHTML = '';
    }, delay);
}

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

    // Очищаем содержимое AI-блока, а не только скрываем
    const aiBlock = document.getElementById('ai-block');
    aiBlock.style.display = 'none';
    aiBlock.innerHTML = '';

    // Сбрасываем режим выделения при очистке библиотеки
    if (typeof window.clearSelection === "function") {
        window.clearSelection();
    }

    showNotification('Запрашиваем список у Steam...', 'info');

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

            showNotification('Отрисовываем игры...', 'info', 0, `Найдено ${total} игр`);
            window.loadedGames.forEach(g => window.addCard(g));

            if (total > 100) {
                showNotification('Список готов', 'success', 0, 'Цены и жанры подгружаются в фоне...');
            } else {
                showNotification('Загрузка завершена', 'success', 0, `Загружено ${total} игр`);
            }

            startBackgroundSync(data.games);
        } else if (data.error) {
            showNotification('Ошибка', 'error', 0, data.error);
            window.currentLoadedTarget = null;
        }
    } catch (e) {
        showNotification('Ошибка соединения', 'error');
        window.currentLoadedTarget = null;
    } finally {
        window.isProcessing = false;
    }
}

/**
 * Фоновая синхронизация цен и жанров
 */
async function startBackgroundSync(rawList) {
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

        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();

            if (done) {
                showNotification('Загрузка завершена', 'success');
                hideNotification(5000);
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

                    const percent = Math.round((processedCount / total) * 100);
                    showNotification('Загрузка цен и скидок', 'progress', percent, `${processedCount} из ${total} игр`);

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
            showNotification('Загрузка остановлена', 'warning');
            hideNotification(3000);
        } else {
            console.error("Ошибка фоновой синхронизации:", e);
            showNotification('Ошибка загрузки', 'error', 0, e.message || 'Проверьте соединение');
            hideNotification(7000);
        }
    }
}

/**
 * Получить рекомендации от AI
 */
async function getAI() {
    if (window.loadedGames.length === 0) {
        showNotification('Сначала загрузите игры', 'warning');
        hideNotification(3000);
        return;
    }

    if (window.aiProcessing) {
        return;
    }

    const block = document.getElementById('ai-block');
    block.style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });

    const customQuery = document.getElementById('ai-search-query')?.value.trim() || '';

    console.log("Отправляем запрос ИИ:", customQuery);

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
            showNotification('Ошибка AI', 'warning', 0, data.content.error);
            hideNotification(5000);
            window.aiProcessing = false;
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

            window.aiProcessing = false;

            try {
                if (typeof window.clearSelection === "function") {
                    window.clearSelection();
                }
            } catch (cleanupError) {
                console.warn("Ошибка интерфейса при очистке (игнорируем):", cleanupError);
            }
        } else {
            showNotification('Нет результатов', 'warning', 0, 'ИИ не смог подобрать игры');
            hideNotification(5000);
            window.aiProcessing = false;
        }
    } catch(e) {
        console.error("Детальная ошибка во фронтенде:", e);
        showNotification('Ошибка AI', 'error', 0, e.message);
        hideNotification(7000);
        window.aiProcessing = false;
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
        showNotification('Введите ссылку на профиль', 'warning');
        hideNotification(3000);
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
            showNotification('Ошибка авторизации', 'error', 0, data.error || 'Проверьте ссылку');
            hideNotification(5000);
        }
    } catch (e) {
        showNotification('Ошибка соединения', 'error', 0, 'Проверьте интернет-соединение');
        hideNotification(5000);
    } finally {
        btn.innerText = "Привязать профиль";
        btn.disabled = false;
    }
}

// Экспортируем функции в глобальную область
window.loadLibrary = loadLibrary;
window.getAI = getAI;
window.loginWithUrl = loginWithUrl;
window.showNotification = showNotification;
window.hideNotification = hideNotification;
