export interface ConfigItemUI {
  label?: string
  hint?: string
  component?: string
  component_props?: Record<string, any>
  options?: Array<Record<string, any> & {
    title?: string
    value?: any
    disabled?: boolean
    disabled_reason?: string
    description?: string
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
