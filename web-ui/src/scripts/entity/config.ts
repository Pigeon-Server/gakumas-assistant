import type { I18nText } from '@/scripts/i18n/types'

export interface ConfigItemUI {
  label?: I18nText
  hint?: I18nText
  component?: string
  component_props?: Record<string, any>
  options?: Array<Record<string, any> & {
    title?: I18nText
    value?: any
    disabled?: boolean
    disabled_reason?: I18nText
    description?: I18nText
  }>
  visible_if?: Record<string, any>
  readonly?: boolean
  resettable?: boolean
  auto_generate?: boolean
  order?: number
}

export interface ConfigItem<T = any> {
  value: T
  default_value: T
  data_type?: string
  verify: string
  use_verify: boolean
  last_modified_time: string
  ui?: ConfigItemUI
}
