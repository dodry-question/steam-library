// theme.js - Переключение темы

/**
 * Инициализация темы при загрузке страницы
 */
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        updateThemeIcon('light');
    } else {
        updateThemeIcon('dark');
    }
}

/**
 * Переключение темы
 */
function toggleTheme() {
    const isLight = document.body.classList.toggle('light-theme');
    const theme = isLight ? 'light' : 'dark';
    localStorage.setItem('theme', theme);
    updateThemeIcon(theme);
}

/**
 * Обновление иконки темы
 */
function updateThemeIcon(theme) {
    const icon = document.getElementById('theme-icon');
    if (icon) {
        icon.innerText = theme === 'light' ? '🌙' : '☀️';
        icon.title = theme === 'light' ? 'Переключить на темную тему' : 'Переключить на светлую тему';
    }
}

// Экспортируем функции
window.initTheme = initTheme;
window.toggleTheme = toggleTheme;
