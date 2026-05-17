/**
 * 本地化文本参数。
 */
export interface I18nParams {
  [key: string]: string | number | boolean | null | undefined | I18nText | I18nParams | Array<string | number | boolean | null | undefined | I18nText | I18nParams>
}

/**
 * 前后端共享的本地化文本结构。
 */
export interface I18nText {
  key: string
  params?: I18nParams
  fallback?: string
}
