import type { I18nText } from '@/scripts/i18n/types'

/** 游戏玩家信息 */
export interface PlayerStatus {
  level: number
  gem: number
  stamina: number
}

/** 游戏状态 */
export interface GameStatus {
  current_location: string
  player: PlayerStatus
}

export interface DeviceStatus {
  available: boolean
  code: string
  message: I18nText | string
}

/** 应用整体状态 */
export interface AppStatus {
  platform: string
  yolo: boolean
  task: string
  current_task?: string
  suspended_task?: string
  device: DeviceStatus
  game: GameStatus
}
