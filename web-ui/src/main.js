/**
 * main.js
 *
 * Bootstraps Vuetify and other plugins then mounts the App`
 */

// Plugins
import {registerPlugins} from '@/plugins'
import autoSave from "@/scripts/directives/auto_save.js"

// Components
import App from './App.vue'

// Composables
import {createApp} from 'vue'

// Styles
import 'unfonts.css'
import {getRandomTheme} from "@/scripts/utils/theme.ts";
import {useAppStore} from "@/stores/app.js";

// Websocket
import {wsService} from '@/scripts/utils/websocket.ts'
import {getWsUrl} from "@/scripts/utils/wsURL.js";
import {WS_ACTION} from "@/scripts/constants.ts";
import message from "@/scripts/utils/message.js";
import dialogs from "@/scripts/utils/dialogs.js";
import { translateKey, translateAny } from "@/scripts/i18n/translate";
import { ensureSystemLocaleListener, syncLocaleWithPreference } from "@/scripts/i18n/localePreference";

const app = createApp(App)

registerPlugins(app)
syncLocaleWithPreference()
ensureSystemLocaleListener()

const theme = getRandomTheme()
const appStore = useAppStore()
appStore.init()
wsService.connect(getWsUrl("/ws"))
wsService.on(WS_ACTION.ShowMessage_Info, data => {
  message.showInfo(translateAny(data.message), data?.close_delay ? data.close_delay * 1000 : 3000)
})
wsService.on(WS_ACTION.ShowMessage_Warning, data => {
  message.showWarning(translateAny(data.message), data?.close_delay ? data.close_delay * 1000 : 3000)
})
wsService.on(WS_ACTION.ShowMessage_Error, data => {
  message.showError(translateAny(data.message), data?.close_delay ? data.close_delay * 1000 : 3000)
})
wsService.on(WS_ACTION.ShowMessage_Success, data => {
  message.showSuccess(translateAny(data.message), data?.close_delay ? data.close_delay * 1000 : 3000)
})
wsService.on(WS_ACTION.TaskExecutionError, data => {
  dialogs.showTaskErrorReportDialog(data).catch(() => ({ reason: "dismissed" }))
})
wsService.onEvent("disconnect", () => {
  message.showWarning(translateKey('websocket.disconnected'))
})
wsService.onEvent("reconnect", () => {
  message.showSuccess(translateKey('websocket.reconnected'))
})
app.config.globalProperties.$theme = theme
app.directive("auto-save", autoSave)

app.mount('#app')

export default app
