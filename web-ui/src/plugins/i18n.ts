import { createI18n } from 'vue-i18n'

import en from '@/assets/lang/en'
import ja from '@/assets/lang/ja'
import zhHans from '@/assets/lang/zhHans'
import zhHant from '@/assets/lang/zhHant'
import { detectSystemLocale, loadSavedLocale, SYSTEM_LOCALE } from '@/scripts/i18n/localeStorage'

/**
 * 语言代码映射。
 */
export const APP_LOCALES = {
  zhHans: 'zh-Hans',
  zhHant: 'zh-Hant',
  en: 'en',
  ja: 'ja',
} as const

export type AppLocale = keyof typeof APP_LOCALES

/**
 * 默认语言。
 */
export const DEFAULT_LOCALE: AppLocale = 'zhHans'

/**
 * 初始化语言。
 */
const initialLocale = loadSavedLocale()
const normalizedInitialLocale: AppLocale = initialLocale === SYSTEM_LOCALE
  ? detectSystemLocale()
  : ((initialLocale in APP_LOCALES ? initialLocale : DEFAULT_LOCALE) as AppLocale)

/**
 * Vue I18n 实例。
 */
const i18n = createI18n({
  legacy: false,
  globalInjection: true,
  locale: normalizedInitialLocale,
  fallbackLocale: DEFAULT_LOCALE,
  messages: {
    zhHans,
    zhHant,
    en,
    ja,
  },
})

export default i18n
