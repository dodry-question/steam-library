// admin.js - Debug utilities

/**
 * Debug authentication
 */
window.auth = async function(key) {
    if (!key) {
        console.log('Usage: auth("key")');
        return;
    }

    try {
        const response = await fetch('/api/auth/verify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ password: key })
        });

        const data = await response.json();

        if (data.success) {
            localStorage.setItem('session_token', data.token);
            console.log('Access granted');
            console.log('Available commands: cfg(1) - enable, cfg(0) - disable, status()');
            await window.status();
        } else {
            console.log('Access denied');
        }
    } catch (error) {
        console.log('Connection error');
    }
};

/**
 * Configuration control
 */
window.cfg = async function(mode) {
    const token = localStorage.getItem('session_token');

    if (!token) {
        console.log('Authentication required: auth("key")');
        return;
    }

    if (mode !== 0 && mode !== 1) {
        console.log('Usage: cfg(1) for enable, cfg(0) for disable');
        return;
    }

    try {
        const response = await fetch('/api/config/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                token: token,
                enabled: mode === 1
            })
        });

        const data = await response.json();

        if (data.success && data.enabled === true) {
            console.log('AI enabled for all users');
        } else if (data.success && data.enabled === false) {
            console.log('AI disabled for public users');
            console.log('AI available only for authenticated sessions');
        } else if (data.error) {
            console.log('Error: ' + data.error);
            if (data.error === 'Доступ запрещен') {
                localStorage.removeItem('session_token');
                console.log('Session expired. Re-authenticate: auth("key")');
            }
        } else {
            console.log('Configuration update failed');
        }
    } catch (error) {
        console.log('Connection error');
    }
};

/**
 * Status check
 */
window.status = async function() {
    const token = localStorage.getItem('session_token');

    if (!token) {
        console.log('Not authenticated');
        console.log('Use: auth("key")');
        return;
    }

    try {
        const response = await fetch('/api/user/settings', {
            method: 'GET',
            headers: {
                'X-Session-Token': token,
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        });

        const data = await response.json();

        if (data.is_admin) {
            console.log('Current status:');
            if (data.ai_enabled) {
                console.log('  AI: enabled for all users');
            } else {
                console.log('  AI: disabled for public users');
                console.log('  AI: available only for authenticated sessions');
            }
        } else {
            console.log('Session invalid');
            localStorage.removeItem('session_token');
            console.log('Re-authenticate: auth("key")');
        }
    } catch (error) {
        console.log('Connection error');
    }
};


