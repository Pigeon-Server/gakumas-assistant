<script setup>
import { computed, ref } from "vue";
import TaskConfigSectionForm from "@/components/lists/config/task_config_section_form.vue";
import { useAppStore } from "@/stores/app.js";
import { TaskStatus } from "@/scripts/constants";
import dialogs from "@/scripts/utils/dialogs.js";
import { getCurrentLocaleTag, translateAny, translateKey } from '@/scripts/i18n/translate'

const app_store = useAppStore();
const props = defineProps({
  modelValue: {
    type: Boolean,
    default: true,
  },
  temporary: {
    type: Boolean,
    default: false,
  },
  disableTransition: {
    type: Boolean,
    default: false,
  },
  width: {
    type: [Number, String],
    default: 400,
  },
})
const emit = defineEmits(["update:modelValue"])

const statusMap = {
  PENDING: {color: "orange", icon: "md:schedule", labelKey: "tasks.status.PENDING"},
  RUNNING: {color: "blue", icon: "md:cached", labelKey: "tasks.status.RUNNING"},
  SUSPENDED: {color: "blue-lighten-1", icon: "md:pause_circle", labelKey: "tasks.status.SUSPENDED"},
  SUCCESS: {color: "green", icon: "md:task_alt", labelKey: "tasks.status.SUCCESS"},
  FAILED: {color: "red", icon: "md:error", labelKey: "tasks.status.FAILED"},
  CANCELED: {color: "grey", icon: "md:cancel", labelKey: "tasks.status.CANCELED"},
  UNKNOWN: {color: "grey", icon: "md:indeterminate_question_box", labelKey: "tasks.status.UNKNOWN"},
}

function normalizeTimestamp(ts) {
  if (!ts || ts <= 0) return null
  // 如果是秒级（10位），转成毫秒
  if (ts < 1e12) {
    ts = ts * 1000
  }
  return ts
}

function formatAbsoluteTime(ts) {
  const t = normalizeTimestamp(ts)
  if (!t) return translateKey('common.notRun')
  return new Intl.DateTimeFormat(getCurrentLocaleTag(), {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(new Date(t))
}

function formatRelativeTime(ts) {
  if (!ts || ts <= 0) return translateKey('common.notRun')
  ts = normalizeTimestamp(ts)
  const now = Date.now()
  const diff = Math.floor((now - ts) / 1000) // 秒差

  if (diff < 5) return translateKey('tasks.relativeTime.justNow')
  if (diff < 60) return translateKey('tasks.relativeTime.secondsAgo', { count: diff })
  if (diff < 3600) return translateKey('tasks.relativeTime.minutesAgo', { count: Math.floor(diff / 60) })
  if (diff < 86400) return translateKey('tasks.relativeTime.hoursAgo', { count: Math.floor(diff / 3600) })
  if (diff < 172800) return translateKey('tasks.relativeTime.yesterday')
  return translateKey('tasks.relativeTime.daysAgo', { count: Math.floor(diff / 86400) })
}

const taskExecutionBlocked = computed(() => (
  app_store.resource_update_status?.required_resources_ready === false
))

const drawerValue = computed({
  get: () => (props.temporary ? props.modelValue : true),
  set: value => emit("update:modelValue", value),
})

const runningTaskName = ref("")
const runningFromTaskName = ref("")
const togglingTaskName = ref("")

function isTaskBusy(taskName) {
  return runningTaskName.value === taskName || runningFromTaskName.value === taskName || togglingTaskName.value === taskName
}

function canRunTaskFrom(task) {
  return !task.manual_only && app_store.status.task === TaskStatus.PENDING
}

function taskConfigSection(taskName) {
  return app_store.config?.[`task__${taskName}`] || null
}

function saveTaskConfig(taskName) {
  return app_store.save_task_config(taskName)
}

async function runTask(taskName) {
  if (isTaskBusy(taskName)) {
    return
  }
  runningTaskName.value = taskName
  try {
    await app_store.run_task(taskName)
  } finally {
    runningTaskName.value = ""
  }
}

async function runTaskFrom(taskName, taskDescription) {
  if (isTaskBusy(taskName)) {
    return
  }
  await dialogs.confirm(
    translateKey('dialogs.runFromTitle'),
    translateKey('dialogs.runFromDescription', {
      task: translateAny(taskDescription),
    }),
  )
  runningFromTaskName.value = taskName
  try {
    await app_store.run_task_from(taskName)
  } finally {
    runningFromTaskName.value = ""
  }
}

async function toggleTask(taskName, enable) {
  if (isTaskBusy(taskName)) {
    return
  }
  togglingTaskName.value = taskName
  try {
    if (enable) {
      await app_store.enable_task(taskName)
    } else {
      await app_store.disable_task(taskName)
    }
  } finally {
    togglingTaskName.value = ""
  }
}
</script>

<template>
  <v-navigation-drawer
    v-model="drawerValue"
    :permanent="!temporary"
    :temporary="temporary"
    :scrim="temporary"
    :width="width"
    :class="['task_drawer', { 'task_drawer--instant': disableTransition }]"
  >
    <v-card class="task_drawer__title_card">
      <div>
        <div class="task_drawer__title">{{ translateKey('tasks.title') }}</div>
        <div v-if="taskExecutionBlocked" class="task_drawer__hint">
          {{ translateKey('tasks.blockedHint') }}
        </div>
      </div>
    </v-card>
    <v-divider/>

    <div class="task_panels">
      <v-expansion-panels variant="accordion">
        <v-expansion-panel
          v-for="(task, task_name) in app_store.task_list"
          :key="task_name"
          elevation="1"
        >
          <v-expansion-panel-title>
            <v-icon
              :color="statusMap[task.status]?.color || 'grey'"
              :icon="statusMap[task.status]?.icon || 'mdi-help-circle'"
              :class="`mr-2 task_${task.status}`"
            />
            <span class="font-medium">{{ translateAny(task.description) }}</span>
            <template v-slot:actions>
              <v-chip
                v-if="task.manual_only"
                variant="outlined"
                color="secondary"
                size="small"
                class="ml-2">
                {{ translateKey('common.manualOnly') }}
              </v-chip>
              <v-chip
                size="small"
                :color="statusMap[task.status]?.color"
                variant="tonal"
                class="ml-2"
              >
                {{ translateKey(statusMap[task.status]?.labelKey || 'tasks.status.UNKNOWN') }}
              </v-chip>
              <v-chip
                v-if="app_store.status.current_task === task_name && app_store.status.task !== TaskStatus.PENDING"
                size="small"
                color="primary"
                variant="tonal"
                class="ml-2"
              >
                {{ translateKey('common.currentTask') }}
              </v-chip>
            </template>
          </v-expansion-panel-title>

          <v-expansion-panel-text>
            <div class="pa-2">
              <p class="text-body-2">{{ translateKey('tasks.taskName') }}<b>{{ task_name }}</b></p>
              <p class="text-body-2">{{ translateKey('tasks.enabled') }}{{ task.enable ? translateKey('common.yes') : translateKey('common.no') }}</p>
              <p class="text-body-2">
                {{ translateKey('tasks.lastRunTime') }}
                <span :title="formatAbsoluteTime(task.last_run_time)">
                  {{ formatRelativeTime(task.last_run_time) }}
                </span>
              </p>

              <div class="task_actions mt-3">
                <v-btn
                  :disabled="task.status === 'RUNNING' || taskExecutionBlocked || isTaskBusy(task_name)"
                  :loading="runningTaskName === task_name"
                  color="primary"
                  variant="outlined"
                  @click="runTask(task_name)"
                >
                  {{ translateKey('tasks.run') }}
                </v-btn>
                <v-btn
                  v-if="!task.manual_only"
                  :disabled="!canRunTaskFrom(task) || taskExecutionBlocked || isTaskBusy(task_name)"
                  :loading="runningFromTaskName === task_name"
                  color="primary"
                  variant="tonal"
                  :title="taskExecutionBlocked ? translateKey('tasks.runBlockedTitle') : (app_store.status.task !== TaskStatus.PENDING ? translateKey('tasks.runBusyTitle') : translateKey('tasks.runFromTitle'))"
                  @click="runTaskFrom(task_name, task.description)"
                >
                  {{ translateKey('tasks.runFrom') }}
                </v-btn>
                <v-btn
                  v-if="task.enable"
                  :disabled="isTaskBusy(task_name)"
                  :loading="togglingTaskName === task_name"
                  color="error"
                  variant="tonal"
                  @click="toggleTask(task_name, false)"
                >
                  {{ translateKey('tasks.disable') }}
                </v-btn>
                <v-btn
                  v-else
                  :disabled="isTaskBusy(task_name)"
                  :loading="togglingTaskName === task_name"
                  color="success"
                  variant="tonal"
                  @click="toggleTask(task_name, true)"
                >
                  {{ translateKey('tasks.enable') }}
                </v-btn>
              </div>
            </div>
            <div v-if="taskConfigSection(task_name)" class="mt-4">
              <h4>{{ translateKey('tasks.settings') }}</h4>
              <TaskConfigSectionForm
                :config="app_store.config"
                :section="taskConfigSection(task_name)"
                :section-name="`task__${task_name}`"
                :save="() => saveTaskConfig(task_name)"
              />
            </div>
          </v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>
    </div>
  </v-navigation-drawer>
</template>


<style scoped>
.task_drawer {
  max-width: 100%;
}

.task_drawer :deep(.v-navigation-drawer__content) {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.task_drawer--instant {
  transition: none !important;
}

.task_drawer--instant :deep(.v-navigation-drawer__content) {
  transition: none !important;
}

.task_drawer__title_card {
  flex: 0 0 auto;
  min-height: 64px;
  display: flex;
  align-items: center;
  padding: 0 18px;
  background: rgb(var(--v-theme-surface));
  color: rgb(var(--v-theme-on-surface));
  border-bottom: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  box-shadow: 0 1px 0 rgba(var(--v-theme-on-surface), 0.02);
}

.task_drawer__title {
  font-size: 1.05rem;
  font-weight: 700;
  line-height: 1.2;
  text-align: left;
}

.task_drawer__hint {
  margin-top: 6px;
  font-size: 0.82rem;
  line-height: 1.4;
  color: rgba(var(--v-theme-on-surface), 0.68);
}

.task_panels {
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
}

.task_actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.task_actions :deep(.v-btn) {
  margin-right: 0 !important;
}

.task_RUNNING {
  animation: spinPause 3s linear infinite running;
}

@keyframes spinPause {
  0% {
    transform: rotate(0deg);
  }
  33.3% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

@media (max-width: 599px) {
  .task_actions :deep(.v-btn) {
    flex: 1 1 calc(50% - 4px);
    min-width: 0;
  }
}
</style>
