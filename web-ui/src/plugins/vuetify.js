/**
 * plugins/vuetify.js
 *
 * Framework documentation: https://vuetifyjs.com`
 */

// Composables
import { createVuetify } from 'vuetify'
import { md3 } from 'vuetify/blueprints'
import { createVueI18nAdapter } from 'vuetify/locale/adapters/vue-i18n'
import { useI18n } from 'vue-i18n'

import { md } from 'vuetify/iconsets/md'
// Styles
import 'vuetify/styles'
// Icon
import 'material-design-icons-iconfont/dist/material-design-icons.css'
import '@mdi/font/css/materialdesignicons.css'
import { aliases, mdi } from "vuetify/iconsets/mdi";
import i18n from '@/plugins/i18n'

const lightTheme = {
  dark: false,
  colors: {
    background: '#f7f8fb',
    surface: '#ffffff',
    'surface-bright': '#ffffff',
    'surface-light': '#f3f5f8',
    'surface-variant': '#e2e7ee',
    'on-background': '#1f252b',
    'on-surface': '#23292f',
    'on-surface-variant': '#5f6870',
    primary: '#2196f3',
    'on-primary': '#ffffff',
    outline: '#cfd6de',
    'outline-variant': '#e6ebf0',
  },
}


// https://vuetifyjs.com/en/introduction/why-vuetify/#feature-guides
export default createVuetify({
  theme: {
    defaultTheme: 'dark',
    themes: {
      light: lightTheme,
      dark: {
        dark: true,
        colors: {
          primary: '#2196f3',
          'on-primary': '#ffffff',
          // 默认 info 是灰色，覆盖为蓝色
          info: '#2196F3',
        },
      },
    },
  },
  blueprint: md3,
  locale: {
    adapter: createVueI18nAdapter({ i18n, useI18n }),
  },
  icons: {
    defaultSet: 'mdi',
    aliases,
    sets: {
      mdi,
      md,
    },
  },
})
