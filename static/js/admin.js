// admin.js - Админ-панель

/**
 * Открыть админ-панель
 */
function openAdminPanel() {
    const adminToken = localStorage.getItem('admin_token');

    if (adminToken) {
        // Проверяем валидность токена
        fetch('/api/admin/status', {
            headers: {
                'X-Admin-Token': adminToken
            }
        })
        .then(res => res.json())
        .then(data => {
            if (data.is_admin) {
                showAdminControls();
                updateAdminStatus(data.ai_enabled);
            } else {
                localStorage.removeItem('admin_token');
                showAdminLogin();
            }
        })
        .catch(() => {
            showAdminLogin();
        });
    } else {
        showAdminLogin();
    }

    document.getElementById('admin-modal').style.display = 'flex';
}

/**
 * Закрыть админ-панель
 */
function closeAdminPanel() {
    document.getElementById('admin-modal').style.display = 'none';
    document.getElementById('admin-login-error').style.display = 'none';
    document.getElementById('admin-password-input').value = '';
}

/**
 * Показать форму входа
 */
function showAdminLogin() {
    document.getElementById('admin-login-section').style.display = 'block';
    document.getElementById('admin-control-section').style.display = 'none';
}

/**
 * Показать панель управления
 */
function showAdminControls() {
    document.getElementById('admin-login-section').style.display = 'none';
    document.getElementById('admin-control-section').style.display = 'block';
}

/**
 * Активация админа
 */
async function activateAdmin() {
    const password = document.getElementById('admin-password-input').value;
    const errorEl = document.getElementById('admin-login-error');
    const btn = document.getElementById('admin-login-btn');

    if (!password) {
        errorEl.innerText = 'Введите пароль';
        errorEl.style.display = 'block';
        return;
    }

    btn.disabled = true;
    btn.innerText = 'Проверка...';

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
            errorEl.style.display = 'none';
            showAdminControls();

            // Получаем текущий статус AI
            const statusResponse = await fetch('/api/admin/status', {
                headers: {
                    'X-Admin-Token': data.token
                }
            });
            const statusData = await statusResponse.json();
            updateAdminStatus(statusData.ai_enabled);
        } else {
            errorEl.innerText = data.error || 'Неверный пароль';
            errorEl.style.display = 'block';
        }
    } catch (error) {
        errorEl.innerText = 'Ошибка подключения к серверу';
        errorEl.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.innerText = 'Войти';
    }
}

/**
 * Переключение доступа к AI
 */
async function toggleAI(enabled) {
    const adminToken = localStorage.getItem('admin_token');

    if (!adminToken) {
        alert('Необходима авторизация');
        showAdminLogin();
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
                enabled: enabled
            })
        });

        const data = await response.json();

        if (data.success) {
            updateAdminStatus(data.enabled);
        } else {
            alert(data.error || 'Ошибка при изменении настроек');
            if (data.error === 'Доступ запрещен') {
                localStorage.removeItem('admin_token');
                showAdminLogin();
            }
        }
    } catch (error) {
        alert('Ошибка подключения к серверу');
    }
}

/**
 * Обновление статуса AI
 */
function updateAdminStatus(enabled) {
    const statusEl = document.getElementById('admin-status');

    if (enabled) {
        statusEl.innerHTML = '<strong style="color: #5cb85c;">✓ AI доступен для всех пользователей</strong><br><span style="font-size: 12px;">Любой может получать рекомендации</span>';
    } else {
        statusEl.innerHTML = '<strong style="color: #d9534f;">✗ AI доступен только админам</strong><br><span style="font-size: 12px;">Обычные пользователи не могут использовать AI</span>';
    }
}

/**
 * Обработка Enter в поле пароля
 */
document.addEventListener('DOMContentLoaded', function() {
    const passwordInput = document.getElementById('admin-password-input');
    if (passwordInput) {
        passwordInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                activateAdmin();
            }
        });
    }
});

// Экспортируем функции в глобальную область
window.openAdminPanel = openAdminPanel;
window.closeAdminPanel = closeAdminPanel;
window.activateAdmin = activateAdmin;
window.toggleAI = toggleAI;
