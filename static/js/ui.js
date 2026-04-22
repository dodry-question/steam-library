// ui.js - Работа с интерфейсом

/**
 * Очистка экрана
 */
function clearList() {
    if(window.isProcessing) return;
    document.querySelector('.search-container').style.display = 'none';

    if (window.syncController) {
        window.syncController.abort();
        window.syncController = null;
    }

    document.getElementById('container').innerHTML = '';
    window.loadedGames = [];
    window.currentLoadedTarget = null;
    window.recommendedHistory = {};
    document.getElementById('copy-btn').style.display = 'none';
    document.getElementById('status-bar').innerText = '';
    document.getElementById('ai-block').style.display = 'none';

    const emptyState = document.getElementById('empty-state');
    if (emptyState) emptyState.style.display = 'flex';
}

/**
 * Копирование списка игр в буфер обмена
 */
async function copyLibrary() {
    if (window.loadedGames.length === 0) return;

    const confirmCopy = confirm(`Вы хотите скопировать список всех загруженных игр (${window.loadedGames.length} шт.) в буфер обмена?`);

    if (confirmCopy) {
        const textToCopy = window.loadedGames.map(game => game.name).join('\n');

        try {
            await navigator.clipboard.writeText(textToCopy);
            alert("Список игр успешно скопирован в буфер обмена!");
        } catch (err) {
            console.error('Ошибка при копировании:', err);
            alert("Не удалось скопировать список. Проверьте разрешения браузера.");
        }
    }
}

/**
 * Живой поиск по играм
 */
function liveSearch() {
    const query = document.getElementById('search-input').value.toLowerCase().trim();
    document.querySelectorAll('.card').forEach(card => {
        const name = card.querySelector('h2').innerText.toLowerCase();
        card.style.display = name.includes(query) ? 'flex' : 'none';
    });
}

/**
 * Сортировка игр
 */
function sortGames(type) {
    const container = document.getElementById('container');
    if (window.loadedGames.length === 0) return;
    container.innerHTML = '';

    window.loadedGames.sort((a, b) => {
        if (type === 'playtime') return b.playtime_forever - a.playtime_forever;
        if (type === 'price') {
            const getPriceValue = (s) => {
                if (!s || s === "Бесплатно") return 0;
                let targetString = s.startsWith("LOCKED|") ? s.split("|")[1] : s;
                if (targetString.includes("Нет в продаже") || targetString.includes("Недоступно")) return -1;
                let clean = targetString.replace(/\s/g, '').replace(',', '.').replace(/[^0-9.]/g, '');
                let num = parseFloat(clean);
                return isNaN(num) ? -1 : num;
            };
            return getPriceValue(b.price_str) - getPriceValue(a.price_str);
        }
    });
    window.loadedGames.forEach(window.addCard);
}

/**
 * Добавление карточки игры
 */
function addCard(game) {
    const div = document.createElement('div');
    div.className = 'card';

    let displayPrice = game.price_str || "";
    let priceAttr = "";
    let extraClass = "";
    let hideDiscount = false;

    if (displayPrice.startsWith("LOCKED|")) {
        const parts = displayPrice.split("|");
        displayPrice = "Недоступно в РФ";
        priceAttr = `data-price="Цена в мире: ${parts[1]}"`;
        extraClass = "region-locked";
        hideDiscount = true;
    } else if (displayPrice.includes("Нет в продаже") || displayPrice.includes("Недоступно")) {
        div.style.borderColor = "#ff444455";
        extraClass = "blocked";
    }

    let hours = Math.round(game.playtime_forever / 60);
    let timeStr = hours > 0 ? `${hours} ч. в игре` : "Не запускалась";
    let discountHtml = (game.discount_percent > 0 && !hideDiscount)
        ? `<div class="discount-badge">-${game.discount_percent}%</div>`
        : '<div></div>';

    let priceClass = "price " + extraClass;
    if (displayPrice === "Бесплатно") priceClass += " free";

    div.innerHTML = `
        <a href="https://store.steampowered.com/app/${game.steam_id}" target="_blank" draggable="false" style="text-decoration:none; color:inherit; flex-grow:1;">
            <img src="${game.image_url}" loading="lazy" draggable="false" onerror="this.src='https://via.placeholder.com/280x130/171a21/c7d5e0?text=No+Image'">
            <div class="info">
                <h2>${game.name}</h2>
                <div class="genres">${game.genres || ""}</div>
                <div class="playtime">${timeStr}</div>
            </div>
        </a>
        <div class="price-block">
            ${discountHtml}
            <div class="${priceClass}" ${priceAttr}>${displayPrice}</div>
        </div>
    `;
    window.setupCardHold(div, game.name);
    document.getElementById('container').appendChild(div);
}

/**
 * Обновление деталей карточки
 */
function updateCardDetails(game) {
    const cardLink = document.querySelector(`.card a[href*="${game.steam_id}"]`);
    if (!cardLink) return;

    const card = cardLink.closest('.card');

    const genreEl = card.querySelector('.genres');
    if (genreEl) genreEl.innerText = game.genres || "";

    const priceBlock = card.querySelector('.price-block');
    let displayPrice = game.price_str;

    if (!displayPrice || displayPrice === "...") displayPrice = "—";

    let extraClass = "";
    let priceAttr = "";

    if (displayPrice.startsWith("LOCKED|")) {
        const parts = displayPrice.split("|");
        displayPrice = "Недоступно в РФ";
        priceAttr = `data-price="Цена в мире: ${parts[1]}"`;
        extraClass = "region-locked";
    }

    let priceClass = "price " + extraClass;
    if (displayPrice === "Бесплатно") priceClass += " free";

    let discountHtml = (game.discount_percent > 0 && !extraClass)
        ? `<div class="discount-badge">-${game.discount_percent}%</div>`
        : '<div></div>';

    priceBlock.innerHTML = `
        ${discountHtml}
        <div class="${priceClass}" ${priceAttr}>${displayPrice}</div>
    `;
}

/**
 * Переключение вида на мобильных
 */
function setMobileView(mode) {
    const gallery = document.getElementById('container');
    const btnDetailed = document.getElementById('view-detailed');
    const btnCompact = document.getElementById('view-compact');

    gallery.style.transition = 'opacity 0.15s ease-in-out';
    gallery.style.opacity = '0';

    setTimeout(() => {
        if (mode === 'compact') {
            gallery.classList.add('compact-mode');
            btnCompact.classList.add('active');
            btnDetailed.classList.remove('active');
        } else {
            gallery.classList.remove('compact-mode');
            btnDetailed.classList.add('active');
            btnCompact.classList.remove('active');
        }

        gallery.style.opacity = '1';
    }, 150);
}

/**
 * Открыть модальное окно авторизации
 */
function openAuthModal() {
    document.getElementById('auth-modal').style.display = 'flex';
}

/**
 * Закрыть модальное окно авторизации
 */
function closeAuthModal() {
    document.getElementById('auth-modal').style.display = 'none';
}

/**
 * Определение часового пояса
 */
function initTimezone() {
    try {
        const tzFull = Intl.DateTimeFormat().resolvedOptions().timeZone;

        const formatter = new Intl.DateTimeFormat('ru-RU', { timeZoneName: 'short' });
        const tzParts = formatter.formatToParts(new Date());
        const tzShortObj = tzParts.find(p => p.type === 'timeZoneName');

        const tzShort = tzShortObj ? tzShortObj.value : tzFull.split('/').pop();

        const tzElement = document.getElementById('user-tz');
        if (tzElement) {
            tzElement.innerText = `🕒 ${tzShort}`;
            tzElement.title = `Системный часовой пояс: ${tzFull}\n`;
        }
    } catch (e) {
        console.error("Не удалось определить часовой пояс:", e);
        document.getElementById('user-tz').style.display = 'none';
    }
}

// Экспортируем функции в глобальную область
window.clearList = clearList;
window.copyLibrary = copyLibrary;
window.liveSearch = liveSearch;
window.sortGames = sortGames;
window.addCard = addCard;
window.updateCardDetails = updateCardDetails;
window.setMobileView = setMobileView;
window.openAuthModal = openAuthModal;
window.closeAuthModal = closeAuthModal;
window.initTimezone = initTimezone;
