/**
 * Fire Detection Module
 * Handles fire detection alerts, voice notifications, and real-time monitoring
 */

class FireDetectionManager {
    constructor(eventId) {
        this.eventId = eventId;
        this.isEnabled = false;
        this.alertCheckInterval = null;
        this.lastAlertId = null;
        this.audioContext = null;
        
        this.init();
    }
    
    init() {
        // Check initial fire detection status
        this.checkFireDetectionStatus();
        
        // Set up toggle handler
        this.setupToggleHandler();
        
        // Initialize Web Speech API for voice alerts
        this.initializeSpeechSynthesis();
    }
    
    async checkFireDetectionStatus() {
        try {
            const response = await fetch(`/api/fire-detection/status/${this.eventId}/`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isEnabled = data.enabled;
                this.updateToggleUI(data.enabled);
                
                if (data.enabled) {
                    this.startAlertMonitoring();
                } else {
                    this.stopAlertMonitoring();
                }
            }
        } catch (error) {
            console.error('Error checking fire detection status:', error);
        }
    }
    
    setupToggleHandler() {
        const toggle = document.getElementById('fire-detection-toggle');
        if (toggle) {
            toggle.addEventListener('change', async (e) => {
                const enabled = e.target.checked;
                await this.toggleFireDetection(enabled);
            });
        }
    }
    
    async toggleFireDetection(enabled) {
        try {
            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
            const formData = new FormData();
            formData.append('enabled', enabled ? 'true' : 'false');
            
            const response = await fetch(`/api/fire-detection/toggle/${this.eventId}/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrfToken
                },
                body: formData
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isEnabled = data.enabled;
                this.updateToggleUI(data.enabled);
                
                if (data.enabled) {
                    this.showNotification('Fire Detection Enabled', 'Fire detection is now active on all cameras', 'success');
                    this.startAlertMonitoring();
                } else {
                    this.showNotification('Fire Detection Disabled', 'Fire detection has been turned off', 'info');
                    this.stopAlertMonitoring();
                }
            } else {
                this.showNotification('Error', data.message || 'Failed to toggle fire detection', 'error');
            }
        } catch (error) {
            console.error('Error toggling fire detection:', error);
            this.showNotification('Error', 'Failed to toggle fire detection', 'error');
        }
    }
    
    updateToggleUI(enabled) {
        const toggle = document.getElementById('fire-detection-toggle');
        if (toggle) {
            toggle.checked = enabled;
        }
        
        const statusBadge = document.getElementById('fire-detection-status');
        if (statusBadge) {
            statusBadge.textContent = enabled ? 'ACTIVE' : 'INACTIVE';
            statusBadge.className = enabled ? 'badge badge-success' : 'badge badge-secondary';
        }
    }
    
    startAlertMonitoring() {
        // Stop existing interval if any
        this.stopAlertMonitoring();
        
        // Check for new alerts every 3 seconds
        this.alertCheckInterval = setInterval(() => {
            this.checkForNewAlerts();
        }, 3000);
        
        // Initial check
        this.checkForNewAlerts();
    }
    
    stopAlertMonitoring() {
        if (this.alertCheckInterval) {
            clearInterval(this.alertCheckInterval);
            this.alertCheckInterval = null;
        }
    }
    
    async checkForNewAlerts() {
        try {
            const response = await fetch(`/api/fire-detection/alerts/${this.eventId}/`);
            const data = await response.json();
            
            if (data.status === 'success' && data.alerts.length > 0) {
                const latestAlert = data.alerts[0];
                
                // Check if this is a new alert
                if (this.lastAlertId !== latestAlert.id) {
                    this.lastAlertId = latestAlert.id;
                    this.handleNewFireAlert(latestAlert);
                }
                
                // Update alerts UI
                this.updateAlertsUI(data.alerts);
            }
        } catch (error) {
            console.error('Error checking for fire alerts:', error);
        }
    }
    
    handleNewFireAlert(alert) {
        console.log('🔥 NEW FIRE ALERT:', alert);
        
        // Play alarm sound
        this.playAlarmSound();
        
        // Speak voice alert
        const message = `Fire detected at ${alert.camera_name}. Confidence ${Math.round(alert.confidence * 100)} percent.`;
        this.speakAlert(message);
        
        // Show visual notification
        this.showFireAlert(alert);
        
        // Flash the page (optional)
        this.flashScreen();
    }
    
    initializeSpeechSynthesis() {
        if ('speechSynthesis' in window) {
            // Ensure voices are loaded
            window.speechSynthesis.getVoices();
        } else {
            console.warn('Speech synthesis not supported in this browser');
        }
    }
    
    speakAlert(message) {
        if ('speechSynthesis' in window) {
            // Cancel any ongoing speech
            window.speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            
            // Try to use a better voice if available
            const voices = window.speechSynthesis.getVoices();
            const englishVoice = voices.find(voice => voice.lang.startsWith('en-'));
            if (englishVoice) {
                utterance.voice = englishVoice;
            }
            
            window.speechSynthesis.speak(utterance);
        }
    }
    
    playAlarmSound() {
        // Create a simple alarm beep using Web Audio API
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        const now = this.audioContext.currentTime;
        
        // Create oscillator for beep sound
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        // Set frequency (higher pitch for urgency)
        oscillator.frequency.value = 880; // A5 note
        oscillator.type = 'sine';
        
        // Set volume envelope
        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(0.3, now + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.5);
        
        oscillator.start(now);
        oscillator.stop(now + 0.5);
        
        // Play multiple beeps
        setTimeout(() => this.playBeep(), 600);
        setTimeout(() => this.playBeep(), 1200);
    }
    
    playBeep() {
        if (!this.audioContext) return;
        
        const now = this.audioContext.currentTime;
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        oscillator.frequency.value = 880;
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(0.3, now + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
        
        oscillator.start(now);
        oscillator.stop(now + 0.3);
    }
    
    showFireAlert(alert) {
        // Create modal or notification for fire alert
        const alertHtml = `
            <div class="fire-alert-modal" id="fire-alert-${alert.id}" style="
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
                color: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(255, 0, 0, 0.5);
                z-index: 10000;
                max-width: 500px;
                width: 90%;
                text-align: center;
                animation: fireAlertPulse 1s infinite;
            ">
                <div style="font-size: 60px; margin-bottom: 15px;">🔥</div>
                <h2 style="margin: 0 0 10px 0; font-weight: bold;">FIRE DETECTED!</h2>
                <p style="font-size: 18px; margin: 10px 0;">
                    <strong>Location:</strong> ${alert.camera_name}<br>
                    <strong>Confidence:</strong> ${Math.round(alert.confidence * 100)}%<br>
                    <strong>Time:</strong> ${new Date(alert.timestamp).toLocaleTimeString()}
                </p>
                ${alert.snapshot_url ? `
                    <img src="${alert.snapshot_url}" alt="Fire Detection" style="
                        width: 100%;
                        max-height: 200px;
                        object-fit: cover;
                        border-radius: 10px;
                        margin: 15px 0;
                        border: 3px solid white;
                    ">
                ` : ''}
                <button onclick="document.getElementById('fire-alert-${alert.id}').remove()" style="
                    background: white;
                    color: #cc0000;
                    border: none;
                    padding: 12px 30px;
                    border-radius: 25px;
                    font-weight: bold;
                    font-size: 16px;
                    cursor: pointer;
                    margin-top: 15px;
                ">ACKNOWLEDGE</button>
            </div>
        `;
        
        // Add CSS animation if not exists
        if (!document.getElementById('fire-alert-styles')) {
            const style = document.createElement('style');
            style.id = 'fire-alert-styles';
            style.textContent = `
                @keyframes fireAlertPulse {
                    0%, 100% { transform: translate(-50%, -50%) scale(1); }
                    50% { transform: translate(-50%, -50%) scale(1.05); }
                }
            `;
            document.head.appendChild(style);
        }
        
        // Add to page
        const alertDiv = document.createElement('div');
        alertDiv.innerHTML = alertHtml;
        document.body.appendChild(alertDiv.firstElementChild);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            const alertElement = document.getElementById(`fire-alert-${alert.id}`);
            if (alertElement) {
                alertElement.remove();
            }
        }, 10000);
    }
    
    flashScreen() {
        // Flash the screen red to draw attention
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 0, 0, 0.3);
            z-index: 9999;
            pointer-events: none;
            animation: flashFade 0.5s ease-out;
        `;
        
        // Add animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes flashFade {
                0% { opacity: 1; }
                100% { opacity: 0; }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(overlay);
        setTimeout(() => overlay.remove(), 500);
    }
    
    updateAlertsUI(alerts) {
        const alertsList = document.getElementById('fire-alerts-list');
        if (!alertsList) return;
        
        if (alerts.length === 0) {
            alertsList.innerHTML = '<p class="text-muted text-center">No fire alerts</p>';
            return;
        }
        
        alertsList.innerHTML = alerts.slice(0, 10).map(alert => `
            <div class="alert alert-danger" style="margin-bottom: 10px;">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <strong>🔥 ${alert.camera_name}</strong><br>
                        <small>${new Date(alert.timestamp).toLocaleString()}</small><br>
                        <small>Confidence: ${Math.round(alert.confidence * 100)}%</small>
                    </div>
                    ${alert.snapshot_url ? `
                        <img src="${alert.snapshot_url}" alt="Fire" style="
                            width: 80px;
                            height: 80px;
                            object-fit: cover;
                            border-radius: 5px;
                        ">
                    ` : ''}
                </div>
            </div>
        `).join('');
    }
    
    showNotification(title, message, type) {
        // Simple notification (you can replace with your preferred notification library)
        const colors = {
            success: '#28a745',
            error: '#dc3545',
            info: '#17a2b8',
            warning: '#ffc107'
        };
        
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${colors[type] || colors.info};
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 10001;
            max-width: 300px;
            animation: slideIn 0.3s ease-out;
        `;
        notification.innerHTML = `
            <strong>${title}</strong><br>
            <small>${message}</small>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Get event ID from page (adjust selector as needed)
    const eventElement = document.querySelector('[data-event-id]');
    if (eventElement) {
        const eventId = eventElement.getAttribute('data-event-id');
        window.fireDetectionManager = new FireDetectionManager(eventId);
    }
});
