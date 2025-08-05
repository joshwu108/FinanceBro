class WebSocketService {
    private ws: WebSocket | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 1000;

    connect(endpoint: string, onMessage: (event: MessageEvent) => void, onError?: (error: Event) => void, onClose?: () => void) {
        const wsUrl = `ws://localhost:8000${endpoint}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.info(`Websocket connected to ${wsUrl}`);
            this.reconnectAttempts = 0;
        };
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage(data);
            } catch (error) {
                console.error('Error parsing websocket message:', error);
            }
        };
        this.ws.onerror = (error) => {
            console.error(`Websocket error: ${error}`);
            if (onError) onError(error);
        };
        this.ws.onclose = () => {
            console.error('Websocket disconnected');
            if (onClose) onClose();
        };
    };

    private attemptReconnect(endpoint: string, onMessage: (event: MessageEvent) => void, onError?: (error: Event) => void) {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                this.connect(endpoint, onMessage, onError);
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    send(message: string) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(message);
        } else {
            console.error('Websocket not connected');
        }
    }
}

export const websocketService = new WebSocketService();
export default websocketService;