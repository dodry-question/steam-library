// app.js - Главный файл приложения

// Глобальные переменные
window.loadedGames = [];
window.isProcessing = false;
window.aiProcessing = false;
window.currentMood = "hidden gems";
window.currentLoadedTarget = null;
window.syncController = null;
window.recommendedHistory = {};

/* --- ЛОГИКА МЕНЮ НАСТРОЕНИЯ --- */
document.addEventListener('DOMContentLoaded', function() {
    // Инициализация часового пояса
    window.initTimezone();

    // Меню настроения
    const moodTrigger = document.getElementById('mood-trigger');
    const moodMenu = document.getElementById('mood-menu');

    if (moodTrigger) {
        moodTrigger.addEventListener('click', function(e) {
            moodMenu.style.display = moodMenu.style.display === 'block' ? 'none' : 'block';
            e.stopPropagation();
        });
    }

    document.querySelectorAll('.mood-item').forEach(item => {
        item.addEventListener('click', function() {
            document.querySelectorAll('.mood-item').forEach(i => i.classList.remove('active'));
            this.classList.add('active');
            window.currentMood = this.getAttribute('data-value');
            document.getElementById('current-mood-emoji').innerText = this.getAttribute('data-emoji');
            moodMenu.style.display = 'none';
        });
    });

    window.addEventListener('click', function() {
        if (moodMenu) moodMenu.style.display = 'none';
    });

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
});
