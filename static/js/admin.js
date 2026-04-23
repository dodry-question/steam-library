// admin.js - Админ-панель

/**
 * Простая команда для активации админа через консоль
 * Использование: admin("13526")
 */
window.admin = async function(password) {
    if (!password) {
        console.log('%c❌ Использование: admin("пароль")', 'color: #ff5c5c; font-size: 14px;');
        return;
    }

    console.log('%c⏳ Проверка пароля...', 'color: #66c0f4; font-size: 14px;');

    try {
        const response = await fetch('/api/admin/activate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ password })
        });

        const data = await response.json();

        if (data.success) {
            localStorage.setItem('admin_token', data.token);
            console.log('%c✅ Админ-доступ активирован!', 'color: #5cb85c; font-size: 16px; font-weight: bold;');
            console.log('%cТеперь используй команды:', 'color: #a0a8b0; font-size: 13px;');
            console.log('%c  ai_on()  - включить AI для всех', 'color: #66c0f4; font-size: 13px;');
            console.log('%c  ai_off() - отключить AI для всех (только админы)', 'color: #66c0f4; font-size: 13px;');
            console.log('%c  ai_status() - проверить текущий статус', 'color: #66c0f4; font-size: 13px;');

            // Показываем текущий статус
            await window.ai_status();
        } else {
            console.log('%c❌ Неверный пароль', 'color: #ff5c5c; font-size: 14px;');
        }
    } catch (error) {
        console.log('%c❌ Ошибка подключения к серверу', 'color: #ff5c5c; font-size: 14px;');
    }
};

/**
 * Включить AI для всех
 */
window.ai_on = async function() {
    const adminToken = localStorage.getItem('admin_token');

    if (!adminToken) {
        console.log('%c❌ Сначала активируй админ-доступ: admin("пароль")', 'color: #ff5c5c; font-size: 14px;');
        return;
    }

    try {
        const response = await fetch('/api/admin/toggle-ai', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                token: adminToken,
                enabled: true
            })
        });

        const data = await response.json();

        if (data.success) {
            console.log('%c✅ AI включен для всех пользователей', 'color: #5cb85c; font-size: 14px; font-weight: bold;');
        } else {
            console.log('%c❌ ' + (data.error || 'Ошибка'), 'color: #ff5c5c; font-size: 14px;');
            if (data.error === 'Доступ запрещен') {
                localStorage.removeItem('admin_token');
                console.log('%c⚠️ Токен устарел. Активируй доступ заново: admin("пароль")', 'color: #ffcc00; font-size: 13px;');
            }
        }
    } catch (error) {
        console.log('%c❌ Ошибка подключения к серверу', 'color: #ff5c5c; font-size: 14px;');
    }
};

/**
 * Отключить AI для всех (только админы)
 */
window.ai_off = async function() {
    const adminToken = localStorage.getItem('admin_token');

    if (!adminToken) {
        console.log('%c❌ Сначала активируй админ-доступ: admin("пароль")', 'color: #ff5c5c; font-size: 14px;');
        return;
    }

    try {
        const response = await fetch('/api/admin/toggle-ai', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                token: adminToken,
                enabled: false
            })
        });

        const data = await response.json();

        if (data.success) {
            console.log('%c✅ AI отключен для обычных пользователей', 'color: #5cb85c; font-size: 14px; font-weight: bold;');
            console.log('%c⚠️ Только админы могут использовать AI', 'color: #ffcc00; font-size: 13px;');
        } else {
            console.log('%c❌ ' + (data.error || 'Ошибка'), 'color: #ff5c5c; font-size: 14px;');
            if (data.error === 'Доступ запрещен') {
                localStorage.removeItem('admin_token');
                console.log('%c⚠️ Токен устарел. Активируй доступ заново: admin("пароль")', 'color: #ffcc00; font-size: 13px;');
            }
        }
    } catch (error) {
        console.log('%c❌ Ошибка подключения к серверу', 'color: #ff5c5c; font-size: 14px;');
    }
};

/**
 * Проверить статус AI
 */
window.ai_status = async function() {
    const adminToken = localStorage.getItem('admin_token');

    if (!adminToken) {
        console.log('%c⚠️ Админ-доступ не активирован', 'color: #ffcc00; font-size: 14px;');
        console.log('%cДля активации используй: admin("пароль")', 'color: #a0a8b0; font-size: 13px;');
        return;
    }

    try {
        const response = await fetch('/api/admin/status', {
            headers: {
                'X-Admin-Token': adminToken
            }
        });

        const data = await response.json();

        if (data.is_admin) {
            console.log('%c📊 Статус AI:', 'color: #66c0f4; font-size: 14px; font-weight: bold;');
            if (data.ai_enabled) {
                console.log('%c  ✅ Включен для всех пользователей', 'color: #5cb85c; font-size: 13px;');
            } else {
                console.log('%c  ❌ Отключен для обычных пользователей', 'color: #ff5c5c; font-size: 13px;');
                console.log('%c  ⚠️ Доступен только админам', 'color: #ffcc00; font-size: 13px;');
            }
        } else {
            console.log('%c❌ Токен недействителен', 'color: #ff5c5c; font-size: 14px;');
            localStorage.removeItem('admin_token');
            console.log('%c⚠️ Активируй доступ заново: admin("пароль")', 'color: #ffcc00; font-size: 13px;');
        }
    } catch (error) {
        console.log('%c❌ Ошибка подключения к серверу', 'color: #ff5c5c; font-size: 14px;');
    }
};

// Приветственное сообщение при загрузке страницы
console.log('%c🔐 Админ-панель Steam Library Manager', 'color: #66c0f4; font-size: 16px; font-weight: bold;');
console.log('%cДля активации используй: admin("пароль")', 'color: #a0a8b0; font-size: 13px;');

