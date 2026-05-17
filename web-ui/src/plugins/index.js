/**
 * plugins/index.js
 *
 * Automatically included in `./src/main.js`
 */

import router from '@/router'
import pinia from '@/stores'
// Plugins
import i18n from '@/plugins/i18n'
import vuetify from './vuetify'

export function registerPlugins (app) {
  app
    .use(i18n)
    .use(vuetify)
    .use(router)
    .use(pinia)
}
