// app.js - Главный файл приложения

// Глобальные переменные
window.loadedGames = [];
window.isProcessing = false;
window.aiProcessing = false;
window.currentLoadedTarget = null;
window.syncController = null;
window.recommendedHistory = {};

/* --- ЛОГИКА МЕНЮ НАСТРОЕНИЯ --- */
document.addEventListener('DOMContentLoaded', function() {
    // Инициализация часового пояса
    window.initTimezone();

    // Поиск по нажатию Enter
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                window.liveSearch();
            }
        });

        searchInput.addEventListener('input', function() {
            if(this.value === '') window.liveSearch();
        });
    }

    // Поиск профиля по нажатию Enter в поле "ID друга"
    const friendIdInput = document.getElementById('friend-id');
    if (friendIdInput) {
        friendIdInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                window.loadLibrary(true);
            }
        });
    }

    // AI поиск по нажатию Enter
    const aiSearchInput = document.getElementById('ai-search-query');
    if (aiSearchInput) {
        aiSearchInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                window.getAI();
            }
        });
    }

});
