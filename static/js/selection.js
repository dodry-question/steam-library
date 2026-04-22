// selection.js - Логика выделения игр для AI

let isSelectionMode = false;
let selectedGames = new Set();

/**
 * Настройка долгого нажатия на карточке
 */
function setupCardHold(cardElement, gameName) {
    const link = cardElement.querySelector('a');
    let pressTimer;
    let startX = 0;
    let startY = 0;
    let isDragging = false;
    let justLongPressed = false;

    // 1. УПРАВЛЕНИЕ КЛИКОМ
    link.addEventListener('click', (e) => {
        if (justLongPressed) {
            e.preventDefault();
            return;
        }

        if (e.shiftKey) {
            e.preventDefault();
            document.getSelection().removeAllRanges();
            isSelectionMode = true;
            toggleGameSelection(gameName, cardElement);
            return;
        }

        if (isSelectionMode) {
            e.preventDefault();
            toggleGameSelection(gameName, cardElement);
        }
    });

    // 2. СТАРТ НАЖАТИЯ
    const startPress = (e) => {
        if (e.type === 'mousedown' && e.button !== 0) return;

        isDragging = false;
        justLongPressed = false;

        if (e.touches && e.touches.length > 0) {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        }

        pressTimer = setTimeout(() => {
            if (isDragging) return;

            justLongPressed = true;
            isSelectionMode = true;

            if (navigator.vibrate) navigator.vibrate(50);

            if (!selectedGames.has(gameName)) {
                toggleGameSelection(gameName, cardElement);
            }
            if (window.getSelection) window.getSelection().removeAllRanges();
        }, 500);
    };

    const cancelPress = () => clearTimeout(pressTimer);

    // 3. ОТСЛЕЖИВАНИЕ СКРОЛЛА
    link.addEventListener('touchmove', (e) => {
        if (!e.touches || e.touches.length === 0) return;
        let moveX = Math.abs(e.touches[0].clientX - startX);
        let moveY = Math.abs(e.touches[0].clientY - startY);

        if (moveX > 10 || moveY > 10) {
            isDragging = true;
            cancelPress();
        }
    }, {passive: true});

    // 4. КОНЕЦ НАЖАТИЯ
    const endPress = (e) => {
        cancelPress();
        if (justLongPressed) {
            setTimeout(() => { justLongPressed = false; }, 500);
        }
    };

    link.addEventListener('mousedown', startPress);
    link.addEventListener('touchstart', startPress, {passive: true});

    link.addEventListener('mouseup', endPress);
    link.addEventListener('touchend', endPress);
    link.addEventListener('mouseleave', cancelPress);

    link.addEventListener('dragstart', (e) => e.preventDefault());
    link.addEventListener('contextmenu', (e) => e.preventDefault());
}

/**
 * Переключение выделения игры
 */
function toggleGameSelection(gameName, cardElement) {
    if (selectedGames.has(gameName)) {
        selectedGames.delete(gameName);
        cardElement.classList.remove('selected');
    } else {
        selectedGames.add(gameName);
        cardElement.classList.add('selected');
    }

    if (selectedGames.size === 0) {
        isSelectionMode = false;
    }

    updateAiButtonState();
}

/**
 * Обновление состояния кнопки AI
 */
function updateAiButtonState() {
    const mainAiBtn = document.getElementById('main-ai-btn');
    const aiContainer = document.getElementById('ai-split-container');

    const panelAiBtn = document.getElementById('panel-ai-btn');
    const selectedCount = document.getElementById('selected-count');
    const actionPanel = document.getElementById('ai-action-panel');

    if (selectedCount) {
        selectedCount.innerText = selectedGames.size;
    }

    if (selectedGames.size > 0) {
        isSelectionMode = true;

        if (actionPanel) {
            actionPanel.style.display = 'flex';
        }

        if (panelAiBtn) panelAiBtn.innerHTML = `✨ Искать похожее (${selectedGames.size})`;

        if (mainAiBtn) {
            mainAiBtn.innerHTML = `✨ Искать похожее (${selectedGames.size})`;
        }

        if (aiContainer) {
            aiContainer.classList.add('active-selection');
        }

        document.querySelectorAll('.card').forEach(c => c.classList.add('selectable'));

    } else {
        isSelectionMode = false;

        if (actionPanel) {
            actionPanel.style.display = 'none';
        }

        if (mainAiBtn) {
            mainAiBtn.innerHTML = `✨ AI Совет`;
        }

        if (aiContainer) {
            aiContainer.classList.remove('active-selection');
        }

        document.querySelectorAll('.card').forEach(c => c.classList.remove('selectable'));
    }
}

/**
 * Очистка выделения
 */
function clearSelection() {
    selectedGames.clear();
    isSelectionMode = false;

    const aiPanel = document.getElementById('ai-action-panel');
    if (aiPanel) {
        aiPanel.style.display = 'none';
    }

    document.querySelectorAll('.card.selected').forEach(c => c.classList.remove('selected'));
    document.querySelectorAll('.card.selectable').forEach(c => c.classList.remove('selectable'));

    if (typeof updateAiButtonState === 'function') {
        updateAiButtonState();
    }
}

// Экспортируем в глобальную область
window.isSelectionMode = isSelectionMode;
window.selectedGames = selectedGames;
window.setupCardHold = setupCardHold;
window.toggleGameSelection = toggleGameSelection;
window.updateAiButtonState = updateAiButtonState;
window.clearSelection = clearSelection;
