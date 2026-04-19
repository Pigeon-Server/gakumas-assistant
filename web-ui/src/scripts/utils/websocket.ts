import {WS_ACTION_HEAD, WS_ACTION} from "@/scripts/constants"
import {WsEventPayloads, WsOptions} from "@/scripts/entity/webSocket"


type TextHandler<T = any> = (data: T) => void
type BinaryHandler = (data: ArrayBuffer) => void
type WsEvent = 'connect' | 'disconnect' | 'reconnect'
type WsEventHandler = () => void

class WebSocketService {
  private ws: WebSocket | null = null
  private url = ''
  private options: Required<WsOptions>

  private reconnectTimer: number | null = null
  private heartbeatTimer: number | null = null
  private reconnectCount = 0
  private hasConnectedOnce = false

  private textHandlers = new Map<string, TextHandler[]>()
  private binaryHandlers: BinaryHandler[] = []
  private eventHandlers = new Map<WsEvent, WsEventHandler[]>()

  constructor(options?: WsOptions) {
    this.options = {
      reconnect: true,
      reconnectInterval: 2000,
      maxReconnectInterval: 30000,
      heartbeatInterval: 10000,
      ...options
    }
  }

  connect(url: string) {
    if (this.ws) return

    this.url = url
    this.ws = new WebSocket(url)

    this.ws.binaryType = 'arraybuffer'

    this.ws.onopen = () => {
      console.log('[WS] connected')
      const isReconnect = this.reconnectCount > 0
      this.hasConnectedOnce = true
      this.reconnectCount = 0
      this.startHeartbeat()

      if (isReconnect) {
        this.emit('reconnect')
      } else {
        this.emit('connect')
      }
    }

    this.ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        this.handleText(event.data)
      } else {
        this.handleBinary(event.data)
      }
    }

    this.ws.onclose = () => {
      if (this.hasConnectedOnce) {
        console.warn('[WS] closed')
      }
      this.cleanup()
      if (this.hasConnectedOnce) {
        this.emit('disconnect')
      }
      this.tryReconnect()
    }

    this.ws.onerror = () => {
      if (this.hasConnectedOnce) {
        console.error('[WS] error')
      }
      this.ws?.close()
    }
  }

  disconnect() {
    this.options.reconnect = false
    this.cleanup()
    this.ws?.close()
  }

  send(action: string, data: any = {}) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const formattedAction: string = `${WS_ACTION_HEAD}:${action}`
      this.ws.send(JSON.stringify({ action: formattedAction, data }))
    }
  }

  sendBinary(data: ArrayBuffer | Blob) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data)
    }
  }

  on(action: string, handler: TextHandler) {
    const target_action: string = `${WS_ACTION_HEAD}:${action}`
    const list = this.textHandlers.get(target_action) || []
    list.push(handler)
    this.textHandlers.set(target_action, list)
  }

  off(action: string, handler: TextHandler) {
    const list = this.textHandlers.get(action) || []
    this.textHandlers.set(
      action,
      list.filter(h => h !== handler)
    )
  }

  onEvent(event: WsEvent|WsEvent[], handler: WsEventHandler) {
    const events = Array.isArray(event) ? event : [event]

    events.forEach(e => {
      const list = this.eventHandlers.get(e) || []
      list.push(handler)
      this.eventHandlers.set(e, list)
    })
  }

  offEvent(event: WsEvent | WsEvent[], handler: WsEventHandler) {
    const events = Array.isArray(event) ? event : [event]

    events.forEach(e => {
      const list = this.eventHandlers.get(e) || []
      this.eventHandlers.set(
        e,
        list.filter(h => h !== handler)
      )
    })
  }

  private emit(event: WsEvent) {
    this.eventHandlers.get(event)?.forEach(h => h())
  }

  onBinary(handler: BinaryHandler) {
    this.binaryHandlers.push(handler)
  }

  offBinary(handler: BinaryHandler) {
    this.binaryHandlers = this.binaryHandlers.filter(h => h !== handler)
  }

  private handleText(raw: string) {
    try {
      const msg = JSON.parse(raw)
      this.dispatchText(msg.action, msg.data)
    } catch (e) {
      console.error('[WS] invalid text message', raw)
    }
  }

  private handleBinary(data: Blob | ArrayBuffer) {
    if (data instanceof Blob) {
      data.arrayBuffer().then(buf => {
        this.binaryHandlers.forEach(h => h(buf))
      })
    } else {
      this.binaryHandlers.forEach(h => h(data))
    }
  }

  private dispatchText(action: string, data: any) {
    this.textHandlers.get(action)?.forEach(h => h(data))
  }

  private startHeartbeat() {
    this.heartbeatTimer = window.setInterval(() => {
      this.send(WS_ACTION.Ping)
    }, this.options.heartbeatInterval)
  }

  private tryReconnect() {
    if (!this.options.reconnect) return

    const delay = Math.min(
      this.options.reconnectInterval * Math.pow(2, this.reconnectCount),
      this.options.maxReconnectInterval
    )

    console.log(`[WS] reconnect in ${delay}ms`)

    this.reconnectTimer = window.setTimeout(() => {
      this.ws = null
      this.reconnectCount++
      this.connect(this.url)
    }, delay)
  }

  private cleanup() {
    clearInterval(this.heartbeatTimer!)
    clearTimeout(this.reconnectTimer!)
    this.heartbeatTimer = null
    this.reconnectTimer = null
  }
}

export const wsService = new WebSocketService()
