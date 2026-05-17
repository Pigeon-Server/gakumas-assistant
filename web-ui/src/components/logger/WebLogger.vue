<template>
  <div
    ref="layoutRef"
    class="layout"
    :class="{ 'is-open': open, 'is-resizing': isResizing, 'overlay-mode': overlayMode }"
    :style="layoutStyle"
  >
    <button
      v-if="overlayMode"
      class="logger_scrim"
      :class="{ 'logger_scrim--visible': open }"
      type="button"
      :aria-label="translateKey('logger.close')"
      @click="open = false"
    />

    <button class="open_web_logger" @click="open = !open" :title="open ? translateKey('logger.close') : translateKey('logger.open')">
      <v-icon>{{ open ? 'md:arrow_right' : 'md:arrow_left' }}</v-icon>
    </button>

    <div
      class="logger_panel"
      :class="{
        'is-open': open,
        'resize-ready': open && (resizeEdgeArmed || isResizing),
      }"
      @pointerdown="onPanelPointerDown"
      @pointermove="onPanelPointerMove"
      @pointerleave="onPanelPointerLeave"
    >
      <div class="logger_header">
        <div class="left">
          <span class="title">{{ translateKey('logger.title') }}</span>
        </div>
      </div>
      <div ref="termContainer" class="term-container"/>
    </div>
  </div>
</template>

<script setup>
import {computed, ref, onMounted, onBeforeUnmount, watch, nextTick} from 'vue';
import {Terminal} from 'xterm';
import {FitAddon} from 'xterm-addon-fit';
import {SearchAddon} from 'xterm-addon-search';
import 'xterm/css/xterm.css';

import {wsService} from '@/scripts/utils/websocket.js';
import {WS_ACTION} from '@/scripts/constants.ts';
import { translateKey } from '@/scripts/i18n/translate';

const TOGGLE_WIDTH = 40;
const DEFAULT_PANEL_WIDTH = 420;
const MIN_PANEL_WIDTH = 320;
const MIN_MAIN_WIDTH = 360;
const RESIZE_HIT_AREA = 10;
const OVERLAY_BREAKPOINT = 1400;
const MOBILE_BREAKPOINT = 960;

const layoutRef = ref(null);
const open = ref(false);
const paused = ref(false);
const buffer = ref([]);
const panelWidth = ref(DEFAULT_PANEL_WIDTH);
const parentWidth = ref(typeof window !== 'undefined' ? window.innerWidth : 0);
const resizeEdgeArmed = ref(false);
const isResizing = ref(false);
const overlayMode = computed(() => parentWidth.value > 0 && parentWidth.value < OVERLAY_BREAKPOINT);

let term = null;
let fitAddon = null;
let searchAddon = null;
let resizeObserver = null;
let layoutResizeObserver = null;
let stopResizeDrag = null;
let flushTimer = null;
let fitRaf = 0;
const termContainer = ref(null);

// 最多保留多少行
const MAX_LINES = 20000;

const LEVEL_COLOR = {
  DEBUG: '\x1b[38;5;244m', // 灰
  INFO: '\x1b[38;5;39m',  // 蓝
  WARN: '\x1b[38;5;214m', // 橙
  ERROR: '\x1b[38;5;196m'  // 红
};

const RESET = '\x1b[0m';

const layoutStyle = computed(() => ({
  '--logger-toggle-width': `${TOGGLE_WIDTH}px`,
  '--logger-panel-width': `${overlayMode.value ? getOverlayPanelWidth() : panelWidth.value}px`,
  '--logger-layout-width': `${overlayMode.value ? 0 : (open.value ? panelWidth.value : 0)}px`,
}));

function formatRecord(record) {
  const time = record.time ?? '';
  const level = record.level ?? 'INFO';
  const message = record.message ?? '';

  const color = LEVEL_COLOR[level] || '';

  return (
    `${time}` +
    `${color}[${level}]${RESET} ` +
    `${message}`
  );
}

function flushBuffer() {
  if (!buffer.value.length) return;

  for (const record of buffer.value) {
    term.writeln(formatRecord(record));
  }
  buffer.value = [];
}

function onIncomingLog(record) {
  if (!record) return;

  const line = formatRecord(record);

  if (paused.value) {
    buffer.value.push(record);
    return;
  }

  term.writeln(line);
}

function fitTerminal() {
  nextTick(() => {
    try {
      if (!open.value || !fitAddon || !termContainer.value) return;
      fitAddon.fit();
    } catch (e) { /* ignore */
    }
  });
}

function cancelQueuedFit() {
  if (fitRaf) {
    cancelAnimationFrame(fitRaf);
    fitRaf = 0;
  }
}

function queueTerminalFit(frames = 3) {
  cancelQueuedFit();
  let remaining = frames;

  const step = () => {
    fitTerminal();
    if (!open.value || remaining <= 0) {
      fitRaf = 0;
      return;
    }
    remaining -= 1;
    fitRaf = requestAnimationFrame(step);
  };

  fitRaf = requestAnimationFrame(step);
}

function getMaxPanelWidth() {
  const width = getParentWidth();
  if (!width) return DEFAULT_PANEL_WIDTH;
  return Math.max(MIN_PANEL_WIDTH, width - TOGGLE_WIDTH - MIN_MAIN_WIDTH);
}

function getParentWidth() {
  return layoutRef.value?.parentElement?.clientWidth ?? window.innerWidth ?? 0;
}

function getOverlayPanelWidth() {
  const width = parentWidth.value || getParentWidth();
  const horizontalGap = width < MOBILE_BREAKPOINT ? 12 : 24;
  const availableWidth = Math.max(180, width - horizontalGap);
  const preferredWidth = width < MOBILE_BREAKPOINT
    ? width - 12
    : Math.min(panelWidth.value, 520);
  const minWidth = width < MOBILE_BREAKPOINT ? 220 : 280;

  return Math.min(
    Math.max(preferredWidth, Math.min(minWidth, availableWidth)),
    availableWidth,
  );
}

function clampPanelWidth(width) {
  return Math.min(Math.max(width, MIN_PANEL_WIDTH), getMaxPanelWidth());
}

function syncPanelWidth() {
  panelWidth.value = clampPanelWidth(panelWidth.value);
}

function syncLayoutMetrics() {
  parentWidth.value = getParentWidth();
  syncPanelWidth();
}

function isInResizeZone(event) {
  const rect = event.currentTarget.getBoundingClientRect();
  return event.clientX - rect.left <= RESIZE_HIT_AREA;
}

function onPanelPointerMove(event) {
  if (isResizing.value) return;
  resizeEdgeArmed.value = isInResizeZone(event);
}

function onPanelPointerLeave() {
  if (!isResizing.value) {
    resizeEdgeArmed.value = false;
  }
}

function stopPanelResize() {
  stopResizeDrag?.();
  stopResizeDrag = null;
}

function onPanelPointerDown(event) {
  if (!open.value || event.button !== 0 || !isInResizeZone(event)) return;

  const startX = event.clientX;
  const startWidth = panelWidth.value;

  isResizing.value = true;
  resizeEdgeArmed.value = true;
  document.body.style.userSelect = 'none';
  document.body.style.cursor = 'ew-resize';
  event.preventDefault();

  const onMove = moveEvent => {
    const delta = startX - moveEvent.clientX;
    panelWidth.value = clampPanelWidth(startWidth + delta);
  };

  const stop = () => {
    isResizing.value = false;
    resizeEdgeArmed.value = false;
    document.body.style.userSelect = '';
    document.body.style.cursor = '';
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', stop);
    window.removeEventListener('pointercancel', stop);
  };

  stopResizeDrag = stop;
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', stop);
  window.addEventListener('pointercancel', stop);
}

onMounted(() => {
  syncLayoutMetrics();
  term = new Terminal({
    convertEol: true,
    scrollback: MAX_LINES,
    disableStdin: true,
  });
  fitAddon = new FitAddon();
  searchAddon = new SearchAddon();
  term.loadAddon(fitAddon);
  term.loadAddon(searchAddon);
  term.open(termContainer.value);
  fitAddon.fit();

  resizeObserver = new ResizeObserver(() => {
    fitTerminal();
  });
  resizeObserver.observe(termContainer.value);

  const resizeTarget = layoutRef.value?.parentElement ?? layoutRef.value;
  if (resizeTarget) {
    layoutResizeObserver = new ResizeObserver(() => {
      syncLayoutMetrics();
      if (open.value) {
        queueTerminalFit(2);
      }
    });
    layoutResizeObserver.observe(resizeTarget);
  }

  wsService.on(WS_ACTION.BroadcastLog, record => {
    onIncomingLog(record);
  });

  window.addEventListener('resize', syncLayoutMetrics);
  window.addEventListener('resize', fitTerminal);

  watch(open, async (v) => {
    if (v) {
      syncLayoutMetrics();
      await nextTick();
      queueTerminalFit();
      return;
    }
    cancelQueuedFit();
  });

  // 定时 flush 本地 buffer（如果 paused=false，这将把缓存排空并写入 terminal）
  flushTimer = setInterval(() => {
    if (!paused.value) flushBuffer();
  }, 200);
});

onBeforeUnmount(() => {
  clearInterval(flushTimer);
  cancelQueuedFit();
  stopPanelResize();
  window.removeEventListener('resize', syncLayoutMetrics);
  window.removeEventListener('resize', fitTerminal);
  resizeObserver?.disconnect();
  layoutResizeObserver?.disconnect();
  try {
    term.dispose();
  } catch (e) {
  }
});
</script>

<style scoped>
.layout {
  position: relative;
  flex: 0 0 var(--logger-layout-width);
  width: var(--logger-layout-width);
  min-width: 0;
  height: 100%;
  overflow: visible;
  display: flex;
  align-items: stretch;
  justify-content: flex-end;
  pointer-events: none;
  transition: width 0.25s ease, flex-basis 0.25s ease;
}

.layout.overlay-mode {
  flex-basis: 0;
  width: 0;
}

.layout.is-resizing,
.layout.is-resizing .logger_panel,
.layout.is-resizing .open_web_logger {
  transition: none !important;
}

.logger_scrim {
  position: absolute;
  inset: 0;
  border: 0;
  padding: 0;
  background: rgba(0, 0, 0, 0.35);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.25s ease;
  z-index: 1;
}

.logger_scrim--visible {
  opacity: 1;
  pointer-events: auto;
}

.logger_panel {
  position: relative;
  flex: 0 0 var(--logger-panel-width);
  width: var(--logger-panel-width);
  height: 100%;
  background: rgb(var(--v-theme-surface));
  color: rgba(var(--v-theme-on-surface), 0.92);
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  pointer-events: auto;
  border-left: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  box-shadow: -8px 0 20px rgba(0, 0, 0, 0.16);
  z-index: 2;
  transform: translateX(calc(100% + 8px));
  transition: transform 0.25s ease;
  will-change: transform;
  pointer-events: none;

  .logger_header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(var(--v-theme-on-surface), 0.12);

    .left {
      display: flex;
      align-items: center;
      gap: 12px;

      .title {
        margin: 15px 0 15px 5px;
        font-weight: 700;
        font-size: 20px
      }
    }
  }
}

.logger_panel.is-open {
  transform: translateX(0);
  pointer-events: auto;
}

.layout.overlay-mode .logger_panel {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
}

.logger_panel::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 8px;
  background: linear-gradient(to right, rgba(var(--v-theme-on-surface), 0.08), transparent);
  opacity: 0;
  transition: opacity 0.2s ease;
}

.logger_panel.resize-ready {
  cursor: ew-resize;
}

.logger_panel.resize-ready::before {
  opacity: 1;
}

.term-container {
  flex: 1 1 auto;
  padding: 8px;
  background: rgba(var(--v-theme-on-surface), 0.04);
  border-radius: 6px;
  margin: 10px;
  overflow: hidden;
}

.open_web_logger {
  background: rgba(var(--v-theme-surface), 0.92);
  border: none;
  cursor: pointer;
  font-size: 18px;
  width: var(--logger-toggle-width);
  height: 56px;
  padding: 0;
  border-radius: 8px 0 0 8px;
  color: rgba(var(--v-theme-on-surface), 0.92);
  transition: background 0.2s ease, box-shadow 0.2s ease;
  position: absolute;
  left: calc(var(--logger-toggle-width) * -1);
  top: 50%;
  transform: translateY(-50%);
  z-index: 3;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: auto;
  box-shadow: -4px 0 12px rgba(0, 0, 0, 0.18);
}

.layout.overlay-mode .open_web_logger {
  left: auto;
  right: 0;
}

.open_web_logger:hover {
  background: rgba(var(--v-theme-on-surface), 0.12);
}

@media (max-width: 960px) {
  .logger_panel .logger_header .left .title {
    margin: 12px 0 12px 4px;
    font-size: 18px;
  }

  .term-container {
    margin: 8px;
  }
}
</style>
