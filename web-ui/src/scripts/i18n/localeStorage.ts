import { APP_LOCALES } from '@/plugins/i18n'

/**
 * 语言存储键。
 */
export const LOCALE_STORAGE_KEY = 'gakumas-assistant.web-ui.locale'

/**
 * 跟随系统语言时使用的标记。
 */
export const SYSTEM_LOCALE = 'system'

/**
 * 根据浏览器语言推断应用语言。
 *
 * @returns 推断后的语言代码
 */
export function detectSystemLocale(): keyof typeof APP_LOCALES {
  if (typeof navigator === 'undefined') {
    return 'zhHans'
  }

  const candidates = [navigator.language, ...(navigator.languages || [])]
    .filter(Boolean)
    .map(item => item.toLowerCase())

  for (const candidate of candidates) {
    if (candidate.startsWith('zh-hant') || candidate.startsWith('zh-tw') || candidate.startsWith('zh-hk') || candidate.startsWith('zh-mo')) {
      return 'zhHant'
    }
    if (candidate.startsWith('zh')) {
      return 'zhHans'
    }
    if (candidate.startsWith('ja')) {
      return 'ja'
    }
    if (candidate.startsWith('en')) {
      return 'en'
    }
  }

  return 'zhHans'
}

/**
 * 读取本地持久化语言。
 *
 * @returns 已保存的语言代码或跟随系统标记
 */
export function loadSavedLocale(): string {
  if (typeof window === 'undefined') {
    return SYSTEM_LOCALE
  }
  const locale = window.localStorage.getItem(LOCALE_STORAGE_KEY)
  if (!locale) {
    return SYSTEM_LOCALE
  }
  if (locale === SYSTEM_LOCALE || locale in APP_LOCALES) {
    return locale
  }
  return SYSTEM_LOCALE
}

/**
 * 保存语言代码。
 *
 * @param locale 语言代码
 */
export function saveLocale(locale: string): void {
  if (typeof window === 'undefined') {
    return
  }
  window.localStorage.setItem(LOCALE_STORAGE_KEY, locale)
}
