<script setup>
import { computed, ref } from "vue";
import auto_purchase_setting from "@/components/lists/config/task_settings/auto_purchase_setting.vue"
import auto_contest_setting from "@/components/lists/config/task_settings/auto_contest_setting.vue";
import dispatch_work_setting from "@/components/lists/config/task_settings/dispatch_work_setting.vue";
import auto_enhancement_support_card_setting from "@/components/lists/config/task_settings/auto_enhancement_support_card_setting.vue";
import auto_producer_setting from "@/components/lists/config/task_settings/auto_producer_setting.vue";
import { useAppStore } from "@/stores/app.js";
import { TaskStatus } from "@/scripts/constants";

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

const settingComponents = {
  auto_purchase: auto_purchase_setting,
  auto_contest: auto_contest_setting,
  dispatch_work: dispatch_work_setting,
  auto_enhancement_support_card: auto_enhancement_support_card_setting,
  auto_producer: auto_producer_setting,
}

const statusMap = {
  PENDING: {color: "orange", icon: "md:schedule", label: "等待中"},
  RUNNING: {color: "blue", icon: "md:cached", label: "运行中"},
  SUSPENDED: {color: "blue-lighten-1", icon: "md:pause_circle", label: "挂起中"},
  SUCCESS: {color: "green", icon: "md:task_alt", label: "已完成"},
  FAILED: {color: "red", icon: "md:error", label: "执行错误"},
  CANCELED: {color: "grey", icon: "md:cancel", label: "已取消"},
  UNKNOWN: {color: "grey", icon: "md:indeterminate_question_box", label: "未知状态"},
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
  if (!t) return "未运行"
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(new Date(t))
}

function formatRelativeTime(ts) {
  if (!ts || ts <= 0) return "未运行"
  ts = normalizeTimestamp(ts)
  const now = Date.now()
  const diff = Math.floor((now - ts) / 1000) // 秒差

  if (diff < 5) return `刚刚`
  if (diff < 60) return `${diff}秒前`
  if (diff < 3600) return `${Math.floor(diff / 60)}分钟前`
  if (diff < 86400) return `${Math.floor(diff / 3600)}小时前`
  if (diff < 172800) return "昨天"
  return `${Math.floor(diff / 86400)}天前`
}

const taskExecutionBlocked = computed(() => (
  app_store.resource_update_status?.required_resources_ready === false
))

const drawerValue = computed({
  get: () => (props.temporary ? props.modelValue : true),
  set: value => emit("update:modelValue", value),
})

const runningTaskName = ref("")
const togglingTaskName = ref("")

function isTaskBusy(taskName) {
  return runningTaskName.value === taskName || togglingTaskName.value === taskName
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
        <div class="task_drawer__title">任务列表</div>
        <div v-if="taskExecutionBlocked" class="task_drawer__hint">
          资源下载完成前可先查看任务和配置，暂不可执行。
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
            <span class="font-medium">{{ task.description }}</span>
            <template v-slot:actions>
              <v-chip
                v-if="task.manual_only"
                size="small"
                class="ml-2">
                仅手动
              </v-chip>
              <v-chip
                size="small"
                :color="statusMap[task.status]?.color"
                text-color="white"
                class="ml-2"
              >
                {{ statusMap[task.status]?.label }}
              </v-chip>
              <v-chip
                v-if="app_store.status.current_task === task_name && app_store.status.task !== TaskStatus.PENDING"
                size="small"
                color="primary"
                class="ml-2"
              >
                当前任务
              </v-chip>
            </template>
          </v-expansion-panel-title>

          <v-expansion-panel-text>
            <div class="pa-2">
              <p class="text-body-2">任务名：<b>{{ task_name }}</b></p>
              <p class="text-body-2">启用：{{ task.enable ? "是" : "否" }}</p>
              <p class="text-body-2">
                上次运行时间：
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
                  执行
                </v-btn>
                <v-btn
                  v-if="task.enable"
                  :disabled="isTaskBusy(task_name)"
                  :loading="togglingTaskName === task_name"
                  color="red"
                  variant="tonal"
                  @click="toggleTask(task_name, false)"
                >
                  禁用
                </v-btn>
                <v-btn
                  v-else
                  :disabled="isTaskBusy(task_name)"
                  :loading="togglingTaskName === task_name"
                  color="green"
                  variant="tonal"
                  @click="toggleTask(task_name, true)"
                >
                  启用
                </v-btn>
              </div>
            </div>
            <div v-if="settingComponents[task_name]" class="mt-4">
              <h4>任务设置</h4>
              <component
                :is="settingComponents[task_name]"
                :task="task"
                :task_name="task_name"
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
  min-height: 60px;
  display: flex;
  align-items: center;
  padding: 0 16px;
}

.task_drawer__title {
  font-size: 1.1rem;
  font-weight: 700;
  line-height: 1.2;
  text-align: left;
}

.task_drawer__hint {
  margin-top: 6px;
  font-size: 0.84rem;
  line-height: 1.4;
  color: rgba(0, 0, 0, 0.64);
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
