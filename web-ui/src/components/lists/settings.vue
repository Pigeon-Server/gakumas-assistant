<script setup>
import { computed } from "vue";
import apis from "@/scripts/apis.js";
import ConfigAutoField from "@/components/lists/config/config_auto_field.vue";
import dialogs from "@/scripts/utils/dialogs.js";
import message from "@/scripts/utils/message.js";
import {useAppStore} from "@/stores/app.ts";
import { translateAny, translateKey } from '@/scripts/i18n/translate'

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
const appStore = useAppStore();
const autoSections = ["base", "dmm_player"]

const settingEntries = computed(() => {
  const entries = []
  for (const sectionName of autoSections) {
    const section = appStore.config?.[sectionName]
    if (!section) {
      continue
    }
    for (const [fieldName, item] of Object.entries(section)) {
      if (!item?.ui?.auto_generate) {
        continue
      }
      if (!isVisible(item.ui?.visible_if)) {
        continue
      }
      entries.push({
        key: `${sectionName}.${fieldName}`,
        sectionName,
        fieldName,
        item,
        order: item.ui?.order ?? 0,
      })
    }
  }
  return entries.sort((left, right) => left.order - right.order)
})

const showDmmRefresh = computed(() => appStore.config?.base?.run_mode?.value === "PC")
const showResourceUpdateTools = computed(() => Boolean(appStore.config?.base))
const resourceUpdateBusy = computed(() => Boolean(appStore.resource_update_status?.checking || appStore.resource_update_status?.updating))
const resourceUpdateChecking = computed(() => Boolean(appStore.resource_update_status?.checking && !appStore.resource_update_status?.updating))
const resourceUpdateStatusText = computed(() => appStore.build_resource_update_status_text(appStore.resource_update_status))
const resourceUpdateHasAction = computed(() => {
  const status = appStore.resource_update_status
  return Boolean(status?.bootstrap_required || status?.has_update)
})
const resourceUpdateLastError = computed(() => appStore.resource_update_status?.last_error || "")
const resourceUpdateProgress = computed(() => appStore.resource_update_status?.progress)
const resourceUpdateProgressActive = computed(() => Boolean(resourceUpdateProgress.value?.active))
const resourceUpdateProgressValue = computed(() => {
  const progress = resourceUpdateProgress.value
  if (!progress?.active) {
    return 0
  }
  return progress.bytes_total > 0 ? progress.percent : progress.percent || progress.step_percent
})
const resourceUpdateProgressIndeterminate = computed(() => {
  const progress = resourceUpdateProgress.value
  if (!progress?.active) {
    return false
  }
  return !progress.bytes_total && !(progress.percent > 0 || progress.step_percent > 0)
})
const resourceUpdateActionLabel = computed(() => (
  appStore.resource_update_status?.bootstrap_required
    ? translateKey('settings.resourceUpdate.bootstrapAction')
    : translateKey('settings.resourceUpdate.applyAction')
))
const resourceUpdateCheckLabel = computed(() => (
  resourceUpdateChecking.value
    ? translateKey('settings.resourceUpdate.checkActionBusy')
    : translateKey('settings.resourceUpdate.checkActionIdle')
))
const resourceUpdateStateLabel = computed(() => {
  const status = appStore.resource_update_status
  if (status?.bootstrap_required && !status?.required_resources_ready) {
    return status?.updating
      ? translateKey('settings.resourceUpdate.state.downloading')
      : translateKey('settings.resourceUpdate.state.bootstrapPending')
  }
  if (status?.updating) {
    return translateKey('settings.resourceUpdate.state.updating')
  }
  if (status?.checking) {
    return translateKey('settings.resourceUpdate.state.checking')
  }
  if (status?.has_update) {
    return translateKey('settings.resourceUpdate.state.updateAvailable')
  }
  if (status?.last_error) {
    return translateKey('settings.resourceUpdate.state.error')
  }
  if (status?.last_checked_at) {
    return translateKey('settings.resourceUpdate.state.checked')
  }
  return translateKey('settings.resourceUpdate.state.pending')
})
const resourceUpdateStateColor = computed(() => {
  const status = appStore.resource_update_status
  if (status?.bootstrap_required && !status?.required_resources_ready) {
    return status?.updating ? "primary" : "warning"
  }
  if (status?.updating || status?.checking) {
    return "primary"
  }
  if (status?.has_update) {
    return "warning"
  }
  if (status?.last_error) {
    return "error"
  }
  return "success"
})
const resourceUpdateHeadline = computed(() => {
  const status = appStore.resource_update_status
  if (status?.bootstrap_required && !status?.required_resources_ready) {
    return status?.updating
      ? translateKey('settings.resourceUpdate.headline.bootstrapRunning')
      : translateKey('settings.resourceUpdate.headline.bootstrapIdle')
  }
  if (status?.updating) {
    return translateKey('settings.resourceUpdate.headline.updating')
  }
  if (status?.checking) {
    return translateKey('settings.resourceUpdate.headline.checking')
  }
  if (status?.has_update) {
    return translateKey('settings.resourceUpdate.headline.hasUpdate')
  }
  if (status?.last_error) {
    return translateKey('settings.resourceUpdate.headline.lastError')
  }
  if (status?.last_checked_at) {
    return translateKey('settings.resourceUpdate.headline.checked')
  }
  return translateKey('settings.resourceUpdate.headline.idle')
})
const resourceUpdateNoticeClass = computed(() => {
  switch (appStore.resource_update_latest_event_type) {
    case "success":
      return "resource-update-panel__notice resource-update-panel__notice--success"
    case "warning":
      return "resource-update-panel__notice resource-update-panel__notice--warning"
    default:
      return "resource-update-panel__notice resource-update-panel__notice--info"
  }
})

function getConfigValue(path) {
  const [sectionName, fieldName] = path.split(".")
  return appStore.config?.[sectionName]?.[fieldName]?.value
}

function isVisible(visibleIf) {
  if (!visibleIf) {
    return true
  }
  if (Array.isArray(visibleIf.__or__)) {
    return visibleIf.__or__.some(rule => isVisible(rule))
  }
  if (Array.isArray(visibleIf.__and__)) {
    return visibleIf.__and__.every(rule => isVisible(rule))
  }
  return Object.entries(visibleIf).every(([path, expected]) => {
    const currentValue = getConfigValue(path)
    if (Array.isArray(expected)) {
      return expected.includes(currentValue)
    }
    return currentValue === expected
  })
}

function shouldShowDivider(entryKey) {
  return entryKey === "base.enabled_auto_startup" || entryKey === "base.enabled_check_resource_updates"
}

async function refreshDmmPlayerToken() {
  await apis.refresh_ddm_player_token()
  await appStore.load_config()
  await message.showSuccess(translateKey('settings.refreshLaunchArgsSuccess'))
}

async function checkResourceUpdates() {
  await appStore.check_resource_updates()
}

async function applyResourceUpdates() {
  await appStore.apply_resource_updates()
}

async function triggerResourceAction() {
  if (appStore.resource_update_status?.bootstrap_required) {
    await appStore.start_required_resource_download()
    return
  }
  await applyResourceUpdates()
}

function reset() {
  dialogs.confirm(
    translateKey('dialogs.resetSettingsTitle'),
    translateKey('dialogs.resetSettingsText'),
  ).then(() => {
    appStore.reset_config();
  }).catch(() => {
    console.debug("reset settings dialog dismissed")
  })
}

const drawerValue = computed({
  get: () => (props.temporary ? props.modelValue : true),
  set: value => emit("update:modelValue", value),
})
</script>

<template>
  <v-navigation-drawer
    v-model="drawerValue"
    :permanent="!temporary"
    :temporary="temporary"
    :scrim="temporary"
    :width="width"
    :class="['settings_drawer', { 'settings_drawer--instant': disableTransition }]"
  >
    <v-card class="settings_drawer__title_card">
      <div class="settings_drawer__title">{{ translateKey('settings.title') }}</div>
    </v-card>
    <v-divider/>
    <v-list nav>
      <v-list-item :subtitle="translateKey('settings.basicSection')"/>
      <template v-for="entry in settingEntries" :key="entry.key">
        <v-divider v-if="shouldShowDivider(entry.key)" />
        <ConfigAutoField :config="appStore.config" :item="entry.item" />
      </template>
      <v-list-item v-if="showResourceUpdateTools" class="resource-update-list-item">
        <div class="resource-update-panel">
            <div class="resource-update-panel__header">
              <div class="resource-update-panel__header-main">
              <div class="resource-update-panel__title">{{ translateKey('settings.resourceUpdate.title') }}</div>
              <div class="resource-update-panel__headline">{{ resourceUpdateHeadline }}</div>
            </div>
            <v-chip
              class="resource-update-panel__state"
              :color="resourceUpdateStateColor"
              variant="tonal"
              size="small"
            >
              {{ resourceUpdateStateLabel }}
            </v-chip>
          </div>
          <div class="resource-update-panel__meta">{{ resourceUpdateStatusText }}</div>
          <div
            v-if="appStore.resource_update_status?.bootstrap_required && !appStore.resource_update_status?.required_resources_ready"
            class="resource-update-panel__notice resource-update-panel__notice--info"
          >
            {{ translateKey('settings.resourceUpdate.bootstrapNotice') }}
          </div>
          <div v-if="resourceUpdateProgressActive" class="resource-update-panel__progress">
            <div class="resource-update-panel__progress-head">
              <span>{{ translateAny(resourceUpdateProgress?.title) || translateKey('settings.resourceUpdate.progressFallbackTitle') }}</span>
              <span>{{ resourceUpdateProgressValue.toFixed(1) }}%</span>
            </div>
            <v-progress-linear
              :model-value="resourceUpdateProgressValue"
              :indeterminate="resourceUpdateProgressIndeterminate"
              color="primary"
              rounded
              height="10"
            />
            <div class="resource-update-panel__progress-meta">
              {{ translateAny(resourceUpdateProgress?.message) || resourceUpdateStatusText }}
            </div>
          </div>
          <div
            v-if="resourceUpdateLastError"
            class="resource-update-panel__notice resource-update-panel__notice--warning"
          >
            {{ translateKey('settings.resourceUpdate.recentError', { error: translateAny(resourceUpdateLastError) }) }}
          </div>
          <div v-if="appStore.resource_update_latest_event" :class="resourceUpdateNoticeClass">
            {{ appStore.resource_update_latest_event }}
          </div>
          <div class="resource-update-panel__actions">
            <v-btn
              class="resource-update-panel__button"
              variant="outlined"
              prepend-icon="md:manage_search"
              :loading="resourceUpdateBusy && !appStore.resource_update_status?.updating"
              :disabled="resourceUpdateBusy"
              @click="checkResourceUpdates"
            >
              {{ resourceUpdateCheckLabel }}
            </v-btn>
            <v-btn
              v-if="resourceUpdateHasAction"
              class="resource-update-panel__button"
              :color="appStore.resource_update_status?.bootstrap_required ? 'primary' : 'success'"
              variant="tonal"
              :prepend-icon="appStore.resource_update_status?.bootstrap_required ? 'md:download' : 'md:system_update_alt'"
              :loading="Boolean(appStore.resource_update_status?.updating)"
              :disabled="resourceUpdateBusy"
              @click="triggerResourceAction"
            >
              {{ resourceUpdateActionLabel }}
            </v-btn>
          </div>
        </div>
      </v-list-item>
      <v-list-item v-if="showDmmRefresh">
        <v-btn
          block
          append-icon="md:refresh"
          @click="refreshDmmPlayerToken"
        >
          {{ translateKey('settings.refreshLaunchArgs') }}
        </v-btn>
      </v-list-item>
    </v-list>
    <template v-slot:append>
      <div class="settings_actions pa-2 mb-2 mt-2">
        <v-btn
          class="settings_actions__button"
          @click="appStore.save_config()"
          color="success"
          variant="tonal"
          append-icon="md:save"
        >
          {{ translateKey('common.save') }}
        </v-btn>
        <v-btn
          class="settings_actions__button"
          @click="reset()"
          color="warning"
          variant="tonal"
          append-icon="md:restart_alt"
        >
          {{ translateKey('common.reset') }}
        </v-btn>
      </div>
    </template>
  </v-navigation-drawer>
</template>

<style scoped>
.settings_drawer {
  max-width: 100%;
}

.settings_drawer :deep(.v-navigation-drawer__content) {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.settings_drawer--instant {
  transition: none !important;
}

.settings_drawer--instant :deep(.v-navigation-drawer__content) {
  transition: none !important;
}

.settings_drawer :deep(.v-list) {
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
}

.settings_drawer__title_card {
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

.settings_drawer__title {
  font-size: 1.05rem;
  font-weight: 700;
  line-height: 1.2;
  text-align: left;
}

.settings_actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.settings_actions__button {
  flex: 1 1 140px;
  min-width: 0;
}

.resource-update-list-item {
  align-items: stretch;
}

.resource-update-list-item :deep(.v-list-item__content) {
  width: 100%;
}

.resource-update-panel {
  width: 100%;
  display: grid;
  gap: 12px;
  padding: 16px;
  border-radius: 16px;
  background: rgba(var(--v-theme-on-surface), 0.025);
  border: 1px solid rgba(var(--v-theme-on-surface), 0.06);
  box-shadow: 0 8px 20px rgba(31, 37, 43, 0.04);
}

.resource-update-panel__header {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 12px;
}

.resource-update-panel__header-main {
  flex: 1 1 180px;
  min-width: 0;
}

.resource-update-panel__state {
  flex: 0 0 auto;
  margin-left: auto;
}

.resource-update-panel__title {
  font-size: 16px;
  font-weight: 700;
  line-height: 1.2;
}

.resource-update-panel__headline {
  margin-top: 6px;
  color: rgba(var(--v-theme-on-surface), 0.72);
  font-size: 13px;
  line-height: 1.45;
}

.resource-update-panel__meta {
  color: rgba(var(--v-theme-on-surface), 0.86);
  font-size: 13px;
  line-height: 1.5;
  white-space: normal;
  word-break: break-word;
  overflow-wrap: anywhere;
}

.resource-update-panel__notice {
  padding: 10px 12px;
  border-radius: 12px;
  font-size: 13px;
  line-height: 1.45;
  white-space: normal;
  word-break: break-word;
  overflow-wrap: anywhere;
}

.resource-update-panel__notice--success {
  background: rgba(var(--v-theme-success), 0.10);
  color: rgb(var(--v-theme-success));
}

.resource-update-panel__notice--warning {
  background: rgba(var(--v-theme-warning), 0.10);
  color: rgb(var(--v-theme-warning));
}

.resource-update-panel__notice--info {
  background: rgba(var(--v-theme-info), 0.10);
  color: rgb(var(--v-theme-info));
}

.resource-update-panel__progress {
  display: grid;
  gap: 8px;
}

.resource-update-panel__progress-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  color: rgba(var(--v-theme-on-surface), 0.92);
}

.resource-update-panel__progress-meta {
  color: rgba(var(--v-theme-on-surface), 0.72);
  font-size: 12px;
  line-height: 1.5;
}

.resource-update-panel__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.resource-update-panel__button {
  flex: 1 1 140px;
  min-width: 0;
}

@media (max-width: 599px) {
  .settings_actions {
    gap: 8px;
  }

  .resource-update-panel {
    padding: 14px;
    gap: 10px;
  }

  .resource-update-panel__header {
    flex-direction: column;
    align-items: stretch;
  }

  .resource-update-panel__state {
    margin-left: 0;
    align-self: flex-start;
  }

  .resource-update-panel__actions {
    gap: 8px;
  }
}
</style>
