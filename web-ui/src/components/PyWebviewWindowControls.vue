<template>
  <div
    v-if="visible"
    class="window_controls"
  >
    <v-btn
      class="window_controls__button"
      icon="md:minimize"
      variant="text"
      size="small"
      :title="translateKey('app.window.minimize')"
      :disabled="pending"
      @click="minimizeWindow"
    />
    <v-btn
      class="window_controls__button"
      :icon="isMaximized ? 'md:filter_none' : 'md:crop_square'"
      variant="text"
      size="small"
      :title="isMaximized ? translateKey('app.window.restore') : translateKey('app.window.maximize')"
      :disabled="pending"
      @click="toggleMaximizeWindow"
    />
    <v-btn
      class="window_controls__button window_controls__button--close"
      icon="md:close"
      variant="text"
      size="small"
      :title="translateKey('app.window.close')"
      @click="closeWindow"
    />
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { addWindowHostReadyListener, getWindowHostApi } from "@/scripts/utils/windowHost.js";
import { translateKey } from "@/scripts/i18n/translate";

const isReady = ref(false);
const isFrameless = ref(false);
const isMaximized = ref(false);
const pending = ref(false);

const visible = computed(() => isReady.value && isFrameless.value);
let syncTimer = null;
let removeWindowHostReadyListener = null;

function applyState(state) {
  if (!state) {
    return;
  }

  isFrameless.value = !!state.frameless;
  isMaximized.value = !!state.maximized;
}

async function syncWindowState() {
  const api = getWindowHostApi();

  if (!api?.get_window_state) {
    return;
  }

  try {
    const state = await api.get_window_state();
    applyState(state);
    isReady.value = true;
  } catch (error) {
    console.error("Failed to sync window host state.", error);
  }
}

function scheduleWindowStateSync() {
  if (syncTimer) {
    clearTimeout(syncTimer);
  }

  syncTimer = window.setTimeout(() => {
    syncTimer = null;
    syncWindowState();
  }, 80);
}

async function invokeWindowAction(actionName) {
  const api = getWindowHostApi();
  const action = api?.[actionName];

  if (!action || pending.value) {
    return;
  }

  pending.value = true;

  try {
    const state = await action();
    applyState(state);
    isReady.value = true;
  } catch (error) {
    console.error(`Failed to execute window host action "${actionName}".`, error);
  } finally {
    pending.value = false;
  }
}

function minimizeWindow() {
  return invokeWindowAction("minimize_window");
}

function toggleMaximizeWindow() {
  return invokeWindowAction("toggle_maximize_window");
}

function closeWindow() {
  const api = getWindowHostApi();
  api?.close_window?.();
}

onMounted(() => {
  if (getWindowHostApi()) {
    syncWindowState();
  }

  removeWindowHostReadyListener = addWindowHostReadyListener(syncWindowState);
  window.addEventListener("focus", scheduleWindowStateSync);
  window.addEventListener("mouseup", scheduleWindowStateSync);
  window.addEventListener("resize", scheduleWindowStateSync);
});

onBeforeUnmount(() => {
  if (syncTimer) {
    clearTimeout(syncTimer);
  }

  removeWindowHostReadyListener?.();
  removeWindowHostReadyListener = null;
  window.removeEventListener("focus", scheduleWindowStateSync);
  window.removeEventListener("mouseup", scheduleWindowStateSync);
  window.removeEventListener("resize", scheduleWindowStateSync);
});
</script>

<style scoped lang="scss">
.window_controls {
  display: flex;
  align-items: stretch;
  gap: 0;
  align-self: stretch;
  height: 100%;
  margin-left: 4px;
}

.window_controls__button {
  display: grid;
  place-items: center;
  width: 48px;
  min-width: 48px;
  height: 100%;
  min-height: 100%;
  padding: 0;
  border-radius: 0;
  color: rgba(var(--v-theme-on-surface), 0.92);
  transition: background-color 0.16s ease, color 0.16s ease;
}

.window_controls__button :deep(.v-btn__content) {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.window_controls__button :deep(.v-icon) {
  font-size: 20px;
  line-height: 1;
}

.window_controls__button:hover {
  background: rgba(0, 0, 0, 0.12);
}

.window_controls__button :deep(.v-btn__overlay) {
  background: transparent !important;
}

.window_controls__button--close:hover {
  background: rgba(202, 63, 52, 0.95);
  color: white;
}

@media (max-width: 599px) {
  .window_controls__button {
    width: 40px;
    min-width: 40px;
  }
}
</style>
