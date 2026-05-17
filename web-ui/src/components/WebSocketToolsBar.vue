<script setup>
import {computed, ref} from "vue";
import message from "@/scripts/utils/message.js";
import api from "@/scripts/apis.js"
import {useAppStore} from "@/stores/app.js";
import {TaskStatus} from "@/scripts/constants.ts";
import { translateAny, translateKey } from '@/scripts/i18n/translate'

const store = useAppStore();
const startingTaskQueue = ref(false)
const stoppingTaskQueue = ref(false)
const suspendingTask = ref(false)
const resumingTask = ref(false)

const canStartTaskQueue = computed(() => {
  if (store.status.task !== TaskStatus.PENDING || startingTaskQueue.value) {
    return false
  }
  if (store.status.platform === "phone") {
    return true
  }
  return store.status.device.available
})

const canSuspendTask = computed(() => (
  store.status.task === TaskStatus.RUNNING
  && Boolean(store.get_current_task()?.allow_manual_suspend)
  && !suspendingTask.value
))

const canStopTaskQueue = computed(() => (
  store.status.task !== TaskStatus.PENDING
  && !stoppingTaskQueue.value
))

const canResumeTask = computed(() => (
  store.status.task === TaskStatus.SUSPENDED
  && !store.status.current_task
  && Boolean(store.get_suspended_task()?.allow_manual_resume)
  && !resumingTask.value
))

async function startTaskQueue() {
  if (!canStartTaskQueue.value) {
    return
  }
  startingTaskQueue.value = true
  try {
    await api.start_task_queue()
    message.showSuccess(translateKey('toolbar.startQueued'))
  } finally {
    startingTaskQueue.value = false
  }
}

async function suspendTask() {
  if (!canSuspendTask.value) {
    return
  }
  suspendingTask.value = true
  try {
    await api.suspend_task()
    message.showSuccess(translateKey('toolbar.suspendedDone'))
  } finally {
    suspendingTask.value = false
  }
}

async function stopTaskQueue() {
  if (!canStopTaskQueue.value) {
    return
  }
  stoppingTaskQueue.value = true
  try {
    await api.stop_task_queue()
    message.showSuccess(translateKey('toolbar.stopQueued'))
  } finally {
    stoppingTaskQueue.value = false
  }
}

async function resumeTask() {
  if (!canResumeTask.value) {
    return
  }
  resumingTask.value = true
  try {
    await api.resume_task()
    message.showSuccess(translateKey('toolbar.resumedDone'))
  } finally {
    resumingTask.value = false
  }
}
</script>

<template>
  <div class="tools_bar">
    <v-card class="tools_bar__card">
      <v-alert
        v-if="!store.status.device.available"
        class="tools_bar__status_alert"
        :title="translateKey('toolbar.deviceNotReady')"
        :text="store.status.platform === 'phone'
          ? translateKey('toolbar.deviceRetryHint', { message: translateAny(store.status.device.message) })
          : translateAny(store.status.device.message)"
        color="error"
        variant="tonal"
      />
      <v-alert
        v-else-if="store.status.task === TaskStatus.PENDING"
        class="tools_bar__status_alert"
        :title="translateKey('toolbar.waitingAction')"
        color="warning"
      />
      <v-alert
        v-else-if="store.status.task === TaskStatus.RUNNING"
        class="tools_bar__status_alert"
        :title="translateKey('toolbar.running')"
        color="success"
        variant="tonal"
      />
      <v-alert
        v-else-if="store.status.task === TaskStatus.SUSPENDED"
        class="tools_bar__status_alert"
        :title="translateKey('toolbar.suspended')"
        color="info"
        variant="tonal"
      />
      <div class="tools_bar__actions">
        <v-btn @click="startTaskQueue" color="success" variant="tonal" :disabled="!canStartTaskQueue" :loading="startingTaskQueue" v-if="store.status.task === TaskStatus.PENDING">
          {{ translateKey('toolbar.start') }}
        </v-btn>
        <v-btn color="error" variant="tonal" @click="stopTaskQueue" :disabled="!canStopTaskQueue" :loading="stoppingTaskQueue" v-else>
          {{ translateKey('toolbar.stop') }}
        </v-btn>
        <v-btn color="warning" variant="tonal" @click="suspendTask" :disabled="!canSuspendTask" :loading="suspendingTask" v-if="store.status.task === TaskStatus.RUNNING && store.get_current_task()?.allow_manual_suspend">
          {{ translateKey('toolbar.suspend') }}
        </v-btn>
        <v-btn color="success" variant="tonal" @click="resumeTask" :disabled="!canResumeTask" :loading="resumingTask" v-if="store.status.task === TaskStatus.SUSPENDED && !store.status.current_task && store.get_suspended_task()?.allow_manual_resume">
          {{ translateKey('toolbar.resume') }}
        </v-btn>
      </div>
    </v-card>
  </div>
</template>

<style scoped>
.tools_bar {
  flex: 0 0 auto;
  min-width: 0;
}

.tools_bar__card {
  margin-bottom: 30px;
  overflow: hidden;
  background: rgb(var(--v-theme-surface));
  border: 1px solid rgba(var(--v-theme-on-surface), 0.06);
  box-shadow: 0 10px 24px rgba(31, 37, 43, 0.05);
}

.tools_bar__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  padding: 20px 15px;
}

.tools_bar__actions :deep(.v-btn) {
  margin-right: 0 !important;
}

.v-alert {
  padding: 25px 15px;
}

.tools_bar__status_alert {
  margin-bottom: 0;
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
}

@media (max-width: 599px) {
  .tools_bar__card {
    margin-bottom: 20px;
  }

  .tools_bar__actions {
    padding: 16px 12px;
  }

  .tools_bar__actions :deep(.v-btn) {
    flex: 1 1 140px;
    min-width: 0;
  }
}
</style>
