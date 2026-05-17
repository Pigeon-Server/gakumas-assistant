/**
 * 主题模式。
 */
export type ThemePreference = 'system' | 'light' | 'dark'

/**
 * 主题存储键。
 */
export const THEME_STORAGE_KEY = 'gakumas-assistant.web-ui.theme'

/**
 * 读取本地持久化主题模式。
 *
 * @returns 已保存的主题模式
 */
export function loadSavedThemePreference(): ThemePreference | null {
  if (typeof window === 'undefined') {
    return null
  }
  const preference = window.localStorage.getItem(THEME_STORAGE_KEY)
  if (preference === 'system' || preference === 'light' || preference === 'dark') {
    return preference
  }
  return null
}

/**
 * 保存主题模式。
 *
 * @param preference 主题模式
 */
export function saveThemePreference(preference: ThemePreference): void {
  if (typeof window === 'undefined') {
    return
  }
  window.localStorage.setItem(THEME_STORAGE_KEY, preference)
}
