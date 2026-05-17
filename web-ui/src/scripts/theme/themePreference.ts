import type { ThemeInstance } from 'vuetify'
import { computed, ref, watch } from 'vue'

import { loadSavedThemePreference, saveThemePreference, type ThemePreference } from '@/scripts/theme/themeStorage'

const SYSTEM_MEDIA_QUERY = '(prefers-color-scheme: dark)'
const themePreference = ref<ThemePreference>(loadSavedThemePreference() ?? 'system')
const systemPrefersDark = ref(false)
let mediaQueryList: MediaQueryList | null = null
let mediaQueryListenerBound = false
let mediaChangeCleanup: (() => void) | null = null

/**
 * 初始化主题偏好监听。
 */
function ensureMediaQueryListener(): void {
  if (typeof window === 'undefined' || mediaQueryListenerBound) {
    return
  }

  mediaQueryList = window.matchMedia(SYSTEM_MEDIA_QUERY)
  systemPrefersDark.value = mediaQueryList.matches
  const handleMediaChange = (event: MediaQueryListEvent): void => {
    systemPrefersDark.value = event.matches
  }
  mediaQueryList.addEventListener('change', handleMediaChange)
  mediaQueryListenerBound = true
  mediaChangeCleanup = () => {
    mediaQueryList?.removeEventListener('change', handleMediaChange)
    mediaQueryListenerBound = false
  }
}

/**
 * 获取当前生效的 Vuetify 主题名称。
 *
 * @returns 生效主题
 */
export const resolvedThemeName = computed<'light' | 'dark'>(() => {
  if (themePreference.value === 'system') {
    return systemPrefersDark.value ? 'dark' : 'light'
  }
  return themePreference.value
})

/**
 * 使用 Vuetify 主题实例同步主题模式。
 *
 * @param theme Vuetify 主题实例
 * @returns 主题状态与设置方法
 */
export function useThemePreference(theme: ThemeInstance) {
  ensureMediaQueryListener()

  watch(
    resolvedThemeName,
    value => {
      theme.global.name.value = value
    },
    { immediate: true },
  )

  /**
   * 设置主题偏好。
   *
   * @param preference 主题模式
   */
  function setThemePreference(preference: ThemePreference): void {
    themePreference.value = preference
    saveThemePreference(preference)
  }

  return {
    themePreference,
    resolvedThemeName,
    setThemePreference,
  }
}

/**
 * 销毁全局主题监听。
 */
export function cleanupThemePreference(): void {
  mediaChangeCleanup?.()
  mediaChangeCleanup = null
  mediaQueryList = null
}
