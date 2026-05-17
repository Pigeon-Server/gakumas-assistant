import type { I18nText } from '@/scripts/i18n/types'

export interface TaskItem {
  description: I18nText
  enable: boolean
  last_run_time: number
  start_time: number
  status: string
  manual_only: boolean
  allow_manual_suspend: boolean
  allow_manual_resume: boolean
}
