<script setup>
/**
 * 通用卡牌选择对话框组件
 *
 * 支持单选（single）和多选（multi）模式，提供：
 *  - 搜索（名称/翻译/ID）
 *  - 可配置的筛选芯片组
 *  - 无限滚动卡片网格
 *  - 右侧详情面板（通过 slot 自定义）
 *
 * 使用示例：
 *   <CardSelectorDialog
 *     v-model="dialogOpen"
 *     v-model:selected="selectedIds"
 *     :cards="cards"
 *     mode="multi"
 *     title="选择白名单卡牌"
 *     :filter-groups="filterGroups"
 *     :card-image-src="cardImageSrc"
 *     :display-name="displayName"
 *     :card-subtitle="cardSubtitle"
 *     :rarity-color="rarityColor"
 *     :rarity-label="rarityLabel"
 *   >
 *     <template #detail="{ card }">
 *       <!-- 自定义详情内容 -->
 *     </template>
 *   </CardSelectorDialog>
 */
import { computed, ref, watch, onBeforeUnmount, nextTick } from 'vue'
import { translateKey } from '@/scripts/i18n/translate'

const props = defineProps({
  /** 对话框是否可见（v-model） */
  modelValue: { type: Boolean, default: false },
  /** 所有候选卡牌数据 */
  cards: { type: Array, default: () => [] },
  /** 已选中的卡牌 ID 列表（v-model:selected） */
  selected: { type: Array, default: () => [] },
  /** 选择模式：'single' 单选 | 'multi' 多选 */
  mode: { type: String, default: 'multi' },
  /** 对话框标题 */
  title: { type: String, default: '' },
  /** 数据加载中 */
  loading: { type: Boolean, default: false },
  /**
   * 筛选组配置，每组包含：
   *   { label: string, key: string, options: [{value, label, color?}] }
   */
  filterGroups: { type: Array, default: () => [] },
  /** 获取卡牌缩略图 URL。返回 null 则显示占位符 */
  cardImageSrc: { type: Function, default: () => null },
  /** 获取卡牌显示名称 */
  displayName: { type: Function, default: (c) => c.translation?.name || c.name || '' },
  /** 获取卡牌副标题 */
  cardSubtitle: { type: Function, default: () => '' },
  /** 获取卡牌稀有度颜色 */
  rarityColor: { type: Function, default: () => '#888' },
  /** 获取卡牌稀有度标签 */
  rarityLabel: { type: Function, default: () => '' },
  /** 搜索框占位符 */
  searchPlaceholder: { type: String, default: '' },
  /** 确认按钮文本（单选模式） */
  confirmText: { type: String, default: null },
  /** 每批加载数量 */
  batchSize: { type: Number, default: 24 },
})

const emit = defineEmits([
  'update:modelValue',
  'update:selected',
  'confirm',
  'cancel',
  'card-detail',
])

// Search
const search = ref('')

// Filter states: { [key]: selectedValues[] }
const filterStates = ref({})
watch(
  () => props.filterGroups,
  (groups) => {
    const states = {}
    for (const g of groups) {
      states[g.key] = filterStates.value[g.key] || []
    }
    filterStates.value = states
  },
  { immediate: true },
)

// Local selection (copy of model on open, committed on confirm)
const localSelection = ref([])

// Detail card (for sidebar slot)
const detailCard = ref(null)

// Infinite scroll
const displayCount = ref(props.batchSize)
const scrollSentinelRef = ref(null)
let observer = null

// Filtered + searched cards
const filteredCards = computed(() => {
  let result = props.cards
  for (const group of props.filterGroups) {
    const selected = filterStates.value[group.key] || []
    if (selected.length > 0) {
      result = result.filter((c) => selected.includes(c[group.key]))
    }
  }
  if (search.value.trim()) {
    const q = search.value.trim().toLowerCase()
    result = result.filter(
      (c) =>
        (c.name || '').toLowerCase().includes(q) ||
        (c.translation?.name || '').toLowerCase().includes(q) ||
        (c.id || '').toLowerCase().includes(q) ||
        (c.characterName || '').toLowerCase().includes(q),
    )
  }
  return result
})

const visibleCards = computed(() => filteredCards.value.slice(0, displayCount.value))
const hasMore = computed(() => displayCount.value < filteredCards.value.length)

function loadMore() {
  if (hasMore.value) {
    displayCount.value += props.batchSize
  }
}

watch([filterStates, search], () => {
  displayCount.value = props.batchSize
}, { deep: true })

watch(hasMore, (val) => {
  if (val && props.modelValue) {
    nextTick(() => {
      if (scrollSentinelRef.value && observer) {
        observer.observe(scrollSentinelRef.value)
      }
    })
  }
})

function setupObserver() {
  if (observer) observer.disconnect()
  observer = new IntersectionObserver(
    (entries) => {
      if (entries[0]?.isIntersecting) loadMore()
    },
    { rootMargin: '200px' },
  )
  nextTick(() => {
    if (scrollSentinelRef.value) observer.observe(scrollSentinelRef.value)
  })
}

// --- Selection logic ---
function isSelected(card) {
  return localSelection.value.includes(card.id)
}

function toggleCard(card) {
  if (props.mode === 'single') {
    localSelection.value = isSelected(card) ? [] : [card.id]
  } else {
    const idx = localSelection.value.indexOf(card.id)
    if (idx >= 0) {
      localSelection.value.splice(idx, 1)
    } else {
      localSelection.value.push(card.id)
    }
  }
}

function showDetail(card, event) {
  if (event) event.stopPropagation()
  detailCard.value = card
  emit('card-detail', card)
}

// --- Dialog lifecycle ---
function onOpen() {
  localSelection.value = [...(props.selected || [])]
  detailCard.value = null
  displayCount.value = props.batchSize
  // Reset filters
  for (const key of Object.keys(filterStates.value)) {
    filterStates.value[key] = []
  }
  search.value = ''
  nextTick(() => setupObserver())
}

function confirmSelection() {
  emit('update:selected', [...localSelection.value])
  emit('confirm', [...localSelection.value])
  emit('update:modelValue', false)
}

function cancelDialog() {
  emit('cancel')
  emit('update:modelValue', false)
}

function clearAll() {
  localSelection.value = []
}

const resolvedTitle = computed(() => props.title || translateKey('dialogs.selector.chooseCard'))
const resolvedSearchPlaceholder = computed(() => props.searchPlaceholder || translateKey('dialogs.selector.searchPlaceholder'))
const resolvedConfirmText = computed(() => props.confirmText || translateKey('dialogs.selector.confirmSingle'))

// Watch dialog visibility for init
watch(
  () => props.modelValue,
  (val) => {
    if (val) onOpen()
  },
)

onBeforeUnmount(() => {
  if (observer) observer.disconnect()
})

const resolvedConfirmLabel = computed(() => {
  if (props.confirmText) return props.confirmText
  if (props.mode === 'single') {
    return translateKey('dialogs.selector.confirmSingle')
  }
  return translateKey('dialogs.selector.confirmMulti', { count: localSelection.value.length })
})

// Expose detail card for parent
defineExpose({ detailCard })
</script>

<template>
  <v-dialog
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    max-width="1200"
    width="calc(100vw - 32px)"
    scrollable
  >
    <v-card class="card-dialog">
      <v-card-title class="dialog-title">
        <span>{{ resolvedTitle }}</span>
        <v-chip size="small" color="primary" variant="tonal" class="ml-2">
          {{ translateKey('dialogs.selector.selectedCount', { count: localSelection.length }) }}
        </v-chip>
        <v-chip size="small" variant="tonal" class="ml-2">
          {{ translateKey('dialogs.selector.totalCards', { count: filteredCards.length }) }}
        </v-chip>
      </v-card-title>

      <v-card-text class="dialog-body">
        <!-- Search -->
        <v-text-field
          v-model="search"
          prepend-inner-icon="md:search"
          :label="resolvedSearchPlaceholder"
          density="compact"
          variant="outlined"
          clearable
          hide-details
          class="mb-3"
        />

        <!-- Filters -->
        <div class="filter-section" v-if="filterGroups.length > 0">
          <div
            v-for="group in filterGroups"
            :key="group.key"
            class="filter-group"
          >
            <span class="filter-label">{{ group.label }}</span>
            <v-chip-group v-model="filterStates[group.key]" multiple>
              <v-chip
                v-for="opt in group.options"
                :key="opt.value"
                :value="opt.value"
                :color="opt.color"
                variant="outlined"
                filter
                size="small"
              >
                {{ opt.label }}
              </v-chip>
            </v-chip-group>
          </div>
        </div>

        <!-- Extra slot above grid (download bar, etc.) -->
        <slot name="above-grid" />

        <!-- Main + Sidebar layout -->
        <div class="content-layout" :class="{ 'has-detail': detailCard && $slots.detail }">
          <!-- Card grid -->
          <div class="content-main">
            <div class="card-grid">
              <div
                v-for="card in visibleCards"
                :key="card.id"
                class="card-item"
                :class="{
                  'card-item--selected': isSelected(card),
                  'card-item--active': detailCard && detailCard.id === card.id,
                }"
                @click="toggleCard(card)"
                @contextmenu.prevent="showDetail(card, $event)"
              >
                <div class="card-image-wrapper">
                  <v-img
                    v-if="cardImageSrc(card)"
                    :src="cardImageSrc(card)"
                    aspect-ratio="1"
                    cover
                    class="card-image"
                    :lazy-src="cardImageSrc(card)"
                  />
                  <div v-else class="card-image-placeholder">
                    <span class="placeholder-rarity" :style="{ color: rarityColor(card) }">
                      {{ rarityLabel(card) }}
                    </span>
                  </div>
                  <v-icon
                    v-if="isSelected(card)"
                    class="card-check"
                    color="white"
                    size="20"
                  >
                    md:check_circle
                  </v-icon>
                  <div class="card-rarity-badge" :style="{ backgroundColor: rarityColor(card) }">
                    {{ rarityLabel(card) }}
                  </div>
                  <!-- Info button -->
                  <v-btn
                    v-if="$slots.detail"
                    icon
                    size="x-small"
                    variant="text"
                    class="card-info-btn"
                    @click="showDetail(card, $event)"
                  >
                    <v-icon size="16" color="white">md:info_outline</v-icon>
                  </v-btn>
                </div>
                <div class="card-name" :title="displayName(card)">
                  {{ displayName(card) }}
                </div>
                <div class="card-type">
                  {{ cardSubtitle(card) }}
                </div>
              </div>
            </div>

            <!-- Empty state -->
            <div v-if="filteredCards.length === 0 && !loading" class="empty-state">
              {{ translateKey('dialogs.selector.noMatchedCards') }}
            </div>

            <!-- Infinite scroll sentinel -->
            <div ref="scrollSentinelRef" class="scroll-sentinel" v-if="hasMore">
              <v-progress-circular indeterminate size="24" width="2" color="primary" />
            </div>
          </div>

          <!-- Detail sidebar (via slot) -->
          <div class="content-sidebar" v-if="detailCard && $slots.detail">
            <slot
              name="detail"
              :card="detailCard"
              :is-selected="isSelected(detailCard)"
              :toggle="() => toggleCard(detailCard)"
              :close="() => (detailCard = null)"
            />
          </div>
        </div>
      </v-card-text>

      <v-card-actions class="dialog-actions">
          <v-btn v-if="mode === 'multi'" variant="text" size="small" @click="clearAll">
          {{ translateKey('dialogs.selector.clearSelection') }}
        </v-btn>
        <v-spacer />
        <v-btn color="error" variant="text" @click="cancelDialog">{{ translateKey('common.cancel') }}</v-btn>
        <v-btn color="primary" variant="flat" @click="confirmSelection">
          {{ resolvedConfirmLabel }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.dialog-title {
  display: flex;
  align-items: center;
  padding: 16px 20px 8px;
}

.dialog-body {
  padding: 8px 20px 16px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.filter-section {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 12px;
}

.filter-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.filter-label {
  font-size: 0.8rem;
  color: rgba(var(--v-theme-on-surface), 0.6);
  white-space: nowrap;
  min-width: 42px;
}

.content-layout {
  display: flex;
  gap: 16px;
  min-height: 0;
  max-height: calc(70vh - 120px);
}

.content-main {
  flex: 1;
  min-width: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
}

.content-sidebar {
  width: 280px;
  flex-shrink: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
  gap: 10px;
}

.card-item {
  cursor: pointer;
  border-radius: 8px;
  padding: 6px;
  transition: background-color 0.15s, box-shadow 0.15s;
  border: 2px solid transparent;
  position: relative;
}

.card-item:hover {
  background-color: rgba(var(--v-theme-on-surface), 0.05);
}

.card-item--selected {
  border-color: rgb(var(--v-theme-primary));
  background-color: rgba(var(--v-theme-primary), 0.08);
}

.card-item--active {
  box-shadow: 0 0 0 1px rgba(var(--v-theme-on-surface), 0.3);
}

.card-image-wrapper {
  position: relative;
  width: 100%;
  aspect-ratio: 1;
  border-radius: 6px;
  overflow: hidden;
  background: rgba(var(--v-theme-on-surface), 0.05);
}

.card-image {
  width: 100%;
  height: 100%;
}

.card-image-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(var(--v-theme-on-surface), 0.03);
}

.placeholder-rarity {
  font-size: 1.5rem;
  font-weight: 700;
  opacity: 0.5;
}

.card-check {
  position: absolute;
  top: 4px;
  right: 4px;
  background: rgba(var(--v-theme-primary), 0.9);
  border-radius: 50%;
}

.card-rarity-badge {
  position: absolute;
  bottom: 4px;
  left: 4px;
  font-size: 0.65rem;
  font-weight: 700;
  color: #000;
  padding: 1px 6px;
  border-radius: 4px;
  line-height: 1.4;
}

.card-info-btn {
  position: absolute;
  bottom: 2px;
  right: 2px;
  opacity: 0;
  transition: opacity 0.15s;
  background: rgba(0, 0, 0, 0.5) !important;
}

.card-item:hover .card-info-btn {
  opacity: 1;
}

.card-name {
  font-size: 0.75rem;
  margin-top: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: 1.3;
}

.card-type {
  font-size: 0.65rem;
  color: rgba(var(--v-theme-on-surface), 0.5);
  line-height: 1.3;
}

.empty-state {
  text-align: center;
  padding: 32px 0;
  color: rgba(var(--v-theme-on-surface), 0.4);
}

.scroll-sentinel {
  display: flex;
  justify-content: center;
  padding: 16px 0;
}

.dialog-actions {
  padding: 8px 16px 12px;
}

@media (max-width: 700px) {
  .content-layout {
    flex-direction: column;
    max-height: none;
  }

  .content-main {
    flex: 1;
    overflow-y: auto;
    max-height: none;
  }

  .content-sidebar {
    width: 100%;
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 100;
    max-height: 55vh;
    overflow-y: auto;
    overscroll-behavior: contain;
    background: rgb(var(--v-theme-surface));
    border-top: 1px solid rgba(var(--v-theme-on-surface), 0.12);
    box-shadow: 0 -4px 16px rgba(0, 0, 0, 0.4);
    border-radius: 12px 12px 0 0;
    padding: 8px;
    animation: slide-up 0.2s ease-out;
  }

  @keyframes slide-up {
    from {
      transform: translateY(100%);
    }
    to {
      transform: translateY(0);
    }
  }

  .card-grid {
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .filter-group {
    flex-wrap: wrap;
  }
}
</style>
