// WebSocket client for real-time event streaming
const WSClient = {
    ws: null,
    url: null,
    reconnectDelay: 2000,
    maxReconnectDelay: 30000,
    onEvent: null,
    _reconnectTimer: null,

    connect(url) {
        this.url = url || `ws://${window.location.host}/ws/events`;
        this._doConnect();
    },

    _doConnect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) return;

        try {
            this.ws = new WebSocket(this.url);
        } catch {
            this._scheduleReconnect();
            return;
        }

        this.ws.onopen = () => {
            console.log("WS connected");
            this.reconnectDelay = 2000;
            this._updateStatus(true);
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (this.onEvent) this.onEvent(data);
            } catch (e) {
                console.warn("Failed to parse WS message:", e);
            }
        };

        this.ws.onclose = () => {
            console.log("WS disconnected");
            this._updateStatus(false);
            this._scheduleReconnect();
        };

        this.ws.onerror = () => {
            this.ws.close();
        };
    },

    _scheduleReconnect() {
        if (this._reconnectTimer) return;
        this._reconnectTimer = setTimeout(() => {
            this._reconnectTimer = null;
            this._doConnect();
        }, this.reconnectDelay);
        this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxReconnectDelay);
    },

    _updateStatus(connected) {
        const el = document.getElementById("status-indicator");
        if (connected) {
            el.textContent = "Connected";
            el.className = "connected";
        } else {
            el.textContent = "Disconnected";
            el.className = "disconnected";
        }
    },

    disconnect() {
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
};
