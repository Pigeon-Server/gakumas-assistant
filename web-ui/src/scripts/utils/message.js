import dialog_utils from '@/scripts/utils/dialogs.js'
import message_component from '@/components/dialogs/message.vue'
import { translateAny, translateKey } from '@/scripts/i18n/translate'

const messageQueue = []
let showingMessage = false

function enqueueMessage(payload) {
  return new Promise(resolve => {
    messageQueue.push({
      payload,
      resolve,
    })
    void flushMessageQueue()
  })
}

async function flushMessageQueue() {
  if (showingMessage) {
    return
  }
  showingMessage = true

  while (messageQueue.length > 0) {
    const currentMessage = messageQueue.shift()
    if (!currentMessage) {
      continue
    }

    try {
      const result = await dialog_utils.init_Dialog(message_component, currentMessage.payload)
      currentMessage.resolve(result)
    } catch (_error) {
      currentMessage.resolve({ reason: 'dismissed' })
    }
  }

  showingMessage = false
}

function showApiErrorMsg (message, status = null, close_delay = 3000) {
  /**
   * 显示API错误信息
   */
  const statusText = status ? translateKey('api.statusSuffix', { status }) : ''
  return enqueueMessage({
    text: translateKey('api.errorPrefix', {
      message: translateAny(message),
      status: statusText,
    }),
    type: 'error',
    timeout: close_delay,
  })
}

function showSuccess (message, close_delay = 3000) {
  /**
   * 显示成功信息
   */
  return enqueueMessage({
    text: message,
    type: 'success',
    timeout: close_delay,
  })
}

function showInfo (message, close_delay = 3000) {
  /**
   * 显示信息
   */
  return enqueueMessage({
    text: message,
    type: 'info',
    timeout: close_delay,
  })
}

function showWarning (message, close_delay = 3000) {
  /**
   * 显示警告信息
   */
  return enqueueMessage({
    text: message,
    type: 'warning',
    timeout: close_delay,
  })
}

function showError (message, close_delay = 3000) {
  /**
   * 显示错误信息
   */
  return enqueueMessage({
    text: message,
    type: 'error',
    timeout: close_delay,
  })
}

export default {
  showApiErrorMsg,
  showSuccess,
  showInfo,
  showWarning,
  showError,
}
