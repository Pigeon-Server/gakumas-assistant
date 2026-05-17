import { createApp } from 'vue'
import inputDialog from '@/components/dialogs/inputDialogV2.vue'
import taskErrorReportDialog from '@/components/dialogs/taskErrorReportDialog.vue'
import vuetify from '@/plugins/vuetify'
import i18n from '@/plugins/i18n'
import confirmDialog from "@/components/dialogs/confirmDialog.vue";

async function init_Dialog (component, other_data = {}) {
  return new Promise((resolve, reject) => {
    const mountNode = document.createElement('div')
    let dialogApp = createApp(component, {
      close: () => {
        if (dialogApp) {
          dialogApp.unmount()
          mountNode.remove()
          dialogApp = undefined
          reject('close')
        }
      },
      confirm: res => {
        resolve(res)
        dialogApp?.unmount()
        mountNode.remove()
        dialogApp = undefined
      },
      ...other_data }).use(vuetify)
      .use(i18n)
    document.body.append(mountNode)
    dialogApp.mount(mountNode)
  })
}

async function showInput_Dialog (title = '', label = '', hint = '', type = 'text', persistent = null, value = null, max = null, min = null) {
  /**
   * 展示输入框
   * @param title 输入框模态标题
   * @param label 输入框内部标题
   * @param hint 输入框提示
   * @param type 输入框类型
   * @param value 输入框值
   * @param max 输入框最大值(仅text与number)
   * @param min 输入框最小值(仅text与number)
   */
  return init_Dialog(inputDialog, {
    title,
    label,
    hint,
    type,
    persistent,
    default_value: value,
    max,
    min,
  })
}

async function confirm (
  title,
  text,
  confirm_text=null,
  close_text=null,
  persistent=false
) {
  return init_Dialog(confirmDialog, {
    title: title,
    description: text,
    persistent,
    confirm_text,
    close_text,
  })
}

async function showTaskErrorReportDialog(payload = {}) {
  return init_Dialog(taskErrorReportDialog, payload)
}

export default {
  init_Dialog,
  confirm,
  inputDialog,
  showInput_Dialog,
  showTaskErrorReportDialog,
}
