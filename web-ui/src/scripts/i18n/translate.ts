import i18n, { APP_LOCALES, DEFAULT_LOCALE } from '@/plugins/i18n'
import type { I18nParams, I18nText } from '@/scripts/i18n/types'
import { getLocalePreference, setLocalePreference, type LocalePreference } from '@/scripts/i18n/localePreference'
import { SYSTEM_LOCALE } from '@/scripts/i18n/localeStorage'

/**
 * 支持的语言列表。
 */
export const SUPPORTED_LOCALES = Object.keys(APP_LOCALES) as Array<keyof typeof APP_LOCALES>

/**
 * 将后端传来的本地化结构解析为字符串。
 *
 * @param value 本地化文本结构或纯字符串
 * @returns 当前语言下的文本
 */
export function translateAny(value: I18nText | string | null | undefined): string {
  if (!value) {
    return ''
  }
  if (typeof value === 'string') {
    return value
  }
  return translateKey(value.key, value.params, value.fallback)
}

/**
 * 翻译列表项标题。
 *
 * @param value 列表项或标题值
 * @returns 翻译结果
 */
export function translateOptionTitle(value: unknown): string {
  if (value && typeof value === 'object') {
    const raw = value as { raw?: { title?: I18nText | string }; title?: I18nText | string }
    if (raw.raw?.title !== undefined) {
      return translateAny(raw.raw.title)
    }
    if (raw.title !== undefined) {
      return translateAny(raw.title)
    }
  }
  return translateAny(value as I18nText | string | null | undefined)
}

/**
 * 翻译指定 key。
 *
 * @param key 翻译键
 * @param params 参数
 * @param fallback 兜底文案
 * @returns 翻译结果
 */
export function translateKey(key: string, params?: I18nParams, fallback?: string): string {
  const { t, te } = i18n.global
  const translatedParams = translateParams(params)

  if (te(key)) {
    return t(key, translatedParams)
  }
  return fallback ?? key
}

/**
 * 递归翻译参数中的国际化对象，避免插值时出现 [object Object]。
 *
 * @param params 原始参数
 * @returns 已翻译的参数对象
 */
function translateParams(params?: I18nParams): Record<string, unknown> {
  if (!params) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(params).map(([key, value]) => [key, translateParamValue(value)]),
  )
}

/**
 * 翻译单个参数值。
 *
 * @param value 参数值
 * @returns 翻译后的值
 */
function translateParamValue(value: unknown): unknown {
  if (!value) {
    return value
  }

  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value
  }

  if (Array.isArray(value)) {
    return value.map(item => translateParamValue(item))
  }

  if (typeof value === 'object') {
    const candidate = value as I18nText
    if (typeof candidate.key === 'string') {
      return translateAny(candidate)
    }

    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, translateParamValue(item)]),
    )
  }

  return value
}

/**
 * 获取当前语言。
 *
 * @returns 当前语言代码
 */
export function getCurrentLocale(): string {
  return i18n.global.locale.value
}

/**
 * 获取当前语言偏好。
 *
 * @returns 当前语言偏好
 */
export function getCurrentLocalePreference(): LocalePreference {
  return getLocalePreference()
}

/**
 * 获取当前语言对应的 BCP 47 语言标识。
 *
 * @returns 当前语言标识
 */
export function getCurrentLocaleTag(): string {
  const locale = getCurrentLocale() as keyof typeof APP_LOCALES
  return APP_LOCALES[locale] ?? APP_LOCALES[DEFAULT_LOCALE]
}

/**
 * 按当前语言格式化文本列表。
 *
 * @param items 待拼接的文本列表
 * @param options 列表格式化选项
 * @returns 格式化后的文本
 */
export function formatLocalizedList(
  items: Array<string | null | undefined>,
  options: Intl.ListFormatOptions = { style: 'long', type: 'conjunction' },
): string {
  const normalizedItems = items.filter((item): item is string => Boolean(item))
  if (!normalizedItems.length) {
    return ''
  }
  return new Intl.ListFormat(getCurrentLocaleTag(), options).format(normalizedItems)
}

/**
 * 设置当前语言。
 *
 * @param locale 语言代码
 */
export function setCurrentLocale(locale: string): void {
  if (locale === SYSTEM_LOCALE) {
    setLocalePreference(SYSTEM_LOCALE)
    return
  }
  if (!SUPPORTED_LOCALES.includes(locale as keyof typeof APP_LOCALES)) {
    setLocalePreference(DEFAULT_LOCALE)
    return
  }
  setLocalePreference(locale as keyof typeof APP_LOCALES)
}
