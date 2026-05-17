import { ref } from 'vue'

import i18n, { APP_LOCALES, DEFAULT_LOCALE, type AppLocale } from '@/plugins/i18n'
import {
  detectSystemLocale,
  loadSavedLocale,
  saveLocale,
  SYSTEM_LOCALE,
} from '@/scripts/i18n/localeStorage'

export type LocalePreference = AppLocale | typeof SYSTEM_LOCALE

const initialLocalePreference = loadSavedLocale()
const localePreference = ref<LocalePreference>(
  initialLocalePreference === SYSTEM_LOCALE || initialLocalePreference in APP_LOCALES
    ? (initialLocalePreference as LocalePreference)
    : SYSTEM_LOCALE,
)
let systemLocaleListenerBound = false

/**
 * 绑定系统语言变化监听。
 */
export function ensureSystemLocaleListener(): void {
  if (typeof window === 'undefined' || systemLocaleListenerBound) {
    return
  }

  window.addEventListener('languagechange', () => {
    if (localePreference.value === SYSTEM_LOCALE) {
      syncLocaleWithPreference(SYSTEM_LOCALE)
    }
  })
  systemLocaleListenerBound = true
}

/**
 * 获取当前语言偏好。
 *
 * @returns 当前语言偏好
 */
export function getLocalePreference(): LocalePreference {
  return localePreference.value
}

/**
 * 获取当前生效语言。
 *
 * @returns 当前生效语言
 */
export function getResolvedLocale(): AppLocale {
  const preference = localePreference.value
  if (preference === SYSTEM_LOCALE) {
    return detectSystemLocale()
  }
  if (preference in APP_LOCALES) {
    return preference as AppLocale
  }
  return DEFAULT_LOCALE
}

/**
 * 设置语言偏好。
 *
 * @param locale 语言偏好
 */
export function setLocalePreference(locale: LocalePreference): void {
  localePreference.value = locale
  saveLocale(locale)
  syncLocaleWithPreference(locale)
}

/**
 * 同步语言偏好到 Vue I18n。
 *
 * @param preference 可选语言偏好
 */
export function syncLocaleWithPreference(preference: LocalePreference = localePreference.value): void {
  const resolvedLocale = preference === SYSTEM_LOCALE
    ? detectSystemLocale()
    : preference
  i18n.global.locale.value = (resolvedLocale in APP_LOCALES ? resolvedLocale : DEFAULT_LOCALE) as AppLocale
}
