<script setup>
import { computed, ref, watch, onBeforeUnmount, nextTick } from 'vue'
import apis from '@/scripts/apis.js'
import { useAppStore } from '@/stores/app.ts'
import { translateAny, translateKey } from '@/scripts/i18n/translate'

const store = useAppStore()

const props = defineProps({
  data: Object,
})

const dialog = ref(false)
const cards = ref([])
const loading = ref(true)
const search = ref('')

// Filter states (arrays: selected values)
const filterRarity = ref([])
const filterType = ref([])
const filterPlan = ref([])

// Local selection (copy of model on open, committed on confirm)
const localSelection = ref([])

// Sidebar detail card
const detailCard = ref(null)
const detailLevel = ref(40)

// 支援発生率 (produceCardUpgradePermil / 10 → %)
function supportTriggerRate(card) {
  const permil = card.produceCardUpgradePermil || 0
  return (permil / 10).toFixed(1)
}

// 根据当前 detailLevel 提取技能描述
function skillDescriptionsAtLevel(card, level) {
  const slots = card.skillSlots || []
  const descs = []
  for (const slot of slots) {
    const levels = slot.levels || []
    let best = null
    for (let i = levels.length - 1; i >= 0; i--) {
      if (levels[i].cardLevel <= level) {
        best = levels[i]
        break
      }
    }
    if (best) {
      descs.push(best.description)
    }
  }
  return descs
}

// 等级上限（从 levelLimits 取满突破最高等级）
function maxLevel(card) {
  if (!card) return 40
  const limits = card.levelLimits || []
  if (limits.length > 0) {
    return Math.max(...limits.map(l => l.levelLimit))
  }
  // 兜底：R=40, SR=50, SSR=60
  if (card.rarity === 'SupportCardRarity_R') return 40
  if (card.rarity === 'SupportCardRarity_Sr') return 50
  return 60
}

// 根据当前 detailLevel 过滤可用的サポートイベント
function eventsAtLevel(card, level) {
  const events = card.events || []
  return events.filter(e => e.supportCardLevel <= level)
}

// 当 detailCard 变化时重置 detailLevel
watch(detailCard, (card) => {
  if (card) {
    detailLevel.value = maxLevel(card)
  }
})

// Infinite scroll
const displayCount = ref(24)
const BATCH_SIZE = 24
const scrollSentinelRef = ref(null)
let observer = null

// 数据库实际枚举值（基于 gakumasu-diff/SupportCard.yaml 分析）
const rarityOptions = computed(() => [
  { value: 'SupportCardRarity_Ssr', label: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.ssr'), color: '#FFD700' },
  { value: 'SupportCardRarity_Sr', label: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.sr'), color: '#C0C0C0' },
  { value: 'SupportCardRarity_R', label: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.r'), color: '#CD7F32' },
])

const typeOptions = computed(() => [
  { value: 'SupportCardType_Vocal', label: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.type.vocal') },
  { value: 'SupportCardType_Dance', label: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.type.dance') },
  { value: 'SupportCardType_Visual', label: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.type.visual') },
  { value: 'SupportCardType_Assist', label: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.type.assist') },
])

// 路线：Plan1=感性, Plan2=逻辑, Plan3=异常, Common=通用
const planOptions = computed(() => [
  { value: 'ProducePlanType_Plan1', label: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.plan.plan1') },
  { value: 'ProducePlanType_Plan2', label: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.plan.plan2') },
  { value: 'ProducePlanType_Plan3', label: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.plan.plan3') },
  { value: 'ProducePlanType_Common', label: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.plan.common') },
])

const rarityOrder = {
  SupportCardRarity_Ssr: 0,
  SupportCardRarity_Sr: 1,
  SupportCardRarity_R: 2,
}

const rarityColor = {
  SupportCardRarity_Ssr: '#FFD700',
  SupportCardRarity_Sr: '#C0C0C0',
  SupportCardRarity_R: '#CD7F32',
}

const rarityLabels = computed(() => ({
  SupportCardRarity_Ssr: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.ssr'),
  SupportCardRarity_Sr: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.sr'),
  SupportCardRarity_R: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.r'),
}))

const typeLabels = computed(() => ({
  SupportCardType_Vocal: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.type.vocal_short'),
  SupportCardType_Dance: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.type.dance_short'),
  SupportCardType_Visual: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.type.visual_short'),
  SupportCardType_Stamina: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.type.stamina_short'),
  SupportCardType_Assist: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.type.assist_short'),
}))

const planLabels = computed(() => ({
  ProducePlanType_Plan1: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.plan.plan1'),
  ProducePlanType_Plan2: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.plan.plan2'),
  ProducePlanType_Plan3: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.plan.plan3'),
  ProducePlanType_Common: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.plan.common'),
}))

const pitemRarityColor = {
  ProduceItemRarity_Ssr: '#FFD700',
  ProduceItemRarity_Sr: '#C0C0C0',
  ProduceItemRarity_R: '#CD7F32',
  ProduceItemRarity_N: '#AAAAAA',
}

const pitemRarityLabels = computed(() => ({
  ProduceItemRarity_Ssr: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.ssr'),
  ProduceItemRarity_Sr: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.sr'),
  ProduceItemRarity_R: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.r'),
  ProduceItemRarity_N: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.n'),
}))

function itemRarityColor(rarity) {
  return pitemRarityColor[rarity] || pcardRarityColor[rarity] || '#AAAAAA'
}

function itemRarityLabel(rarity) {
  return pitemRarityLabels.value[rarity] || pcardRarityLabels.value[rarity] || ''
}

// ProduceCard (skill card) rarity
const pcardRarityColor = {
  ProduceCardRarity_Ssr: '#FFD700',
  ProduceCardRarity_Sr: '#C0C0C0',
  ProduceCardRarity_R: '#CD7F32',
  ProduceCardRarity_N: '#AAAAAA',
}

const pcardRarityLabels = computed(() => ({
  ProduceCardRarity_Ssr: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.ssr'),
  ProduceCardRarity_Sr: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.sr'),
  ProduceCardRarity_R: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.r'),
  ProduceCardRarity_N: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.rarity.n'),
}))

const cardCategoryLabels = computed(() => ({
  ProduceCardCategory_ActiveSkill: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.card_category.active_skill'),
  ProduceCardCategory_MentalSkill: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.card_category.mental_skill'),
  ProduceCardCategory_Trouble: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.card_category.trouble'),
  ProduceCardCategory_FreeSkill: translateKey('backend.config.task__auto_enhancement_support_card.whitelist.card_category.free_skill'),
}))

function eventItemKindLabel(item) {
  if (item.kind === 'card') {
    return cardCategoryLabels.value[item.category] || translateKey('backend.config.task__auto_enhancement_support_card.whitelist.card_category.skill_card')
  }
  return translateKey('backend.config.task__auto_enhancement_support_card.whitelist.card_category.p_item')
}

function eventItemKindColor(item) {
  if (item.kind === 'card') return '#ce93d8' // purple for skill cards
  return '#66bb6a' // green for p-items
}

function eventItemImageSrc(item) {
  if (!item.assetId) return null
  const stripped = item.assetId.replace('img_general_', '')
  if (item.kind === 'card') return `/api/game_assets/skill_cards/${stripped}.png`
  return `/api/game_assets/items/${stripped}.png`
}

function displayName(card) {
  return card.translation?.name || card.name
}

function cardSubtitle(card) {
  const r = rarityLabels.value[card.rarity] || ''
  const t = typeLabels.value[card.type] || ''
  const p = planLabels.value[card.planType] || ''
  return [r, t, p].filter(Boolean).join(' · ')
}

// Filtered + searched cards (search both original name and localized name)
const filteredCards = computed(() => {
  let result = cards.value
  if (filterRarity.value.length > 0) {
    result = result.filter(c => filterRarity.value.includes(c.rarity))
  }
  if (filterType.value.length > 0) {
    result = result.filter(c => filterType.value.includes(c.type))
  }
  if (filterPlan.value.length > 0) {
    result = result.filter(c => filterPlan.value.includes(c.planType))
  }
  if (search.value.trim()) {
    const q = search.value.trim().toLowerCase()
    result = result.filter(c =>
      (c.name || '').toLowerCase().includes(q)
      || (c.translation?.name || '').toLowerCase().includes(q)
      || (c.id || '').toLowerCase().includes(q)
    )
  }
  // 按稀有度排序：SSR > SR > R
  result.sort((a, b) => (rarityOrder[a.rarity] ?? 9) - (rarityOrder[b.rarity] ?? 9))
  return result
})

// Infinite scroll: display a subset of filteredCards
const visibleCards = computed(() => {
  return filteredCards.value.slice(0, displayCount.value)
})

const hasMore = computed(() => displayCount.value < filteredCards.value.length)

function loadMore() {
  if (hasMore.value) {
    displayCount.value += BATCH_SIZE
  }
}

// Reset displayCount when filters change
watch([filterRarity, filterType, filterPlan, search], () => {
  displayCount.value = BATCH_SIZE
})

// Re-observe sentinel when hasMore changes (so observer reconnects after filter changes)
watch(hasMore, (val) => {
  if (val && dialog.value) {
    nextTick(() => {
      if (scrollSentinelRef.value && observer) {
        observer.observe(scrollSentinelRef.value)
      }
    })
  }
})

// IntersectionObserver for infinite scroll
function setupObserver() {
  if (observer) observer.disconnect()
  observer = new IntersectionObserver((entries) => {
    if (entries[0]?.isIntersecting) {
      loadMore()
    }
  }, { rootMargin: '200px' })
  nextTick(() => {
    if (scrollSentinelRef.value) {
      observer.observe(scrollSentinelRef.value)
    }
  })
}

// Selected card objects for display chips
const selectedCards = computed(() => {
  const ids = props.data.whitelist_card_ids.value || []
  return cards.value.filter(c => ids.includes(c.id))
})

function isSelected(card) {
  return localSelection.value.includes(card.id)
}

function toggleCard(card) {
  const idx = localSelection.value.indexOf(card.id)
  if (idx >= 0) {
    localSelection.value.splice(idx, 1)
  } else {
    localSelection.value.push(card.id)
  }
}

function showDetail(card, event) {
  if (event) event.stopPropagation()
  detailCard.value = card
  // 若全尺寸图未缓存，按需触发单卡下载
  if (!card.gameAssetFullImage) {
    triggerSingleCardFullImageDownload(card)
  }
}

function openDialog() {
  localSelection.value = [...(props.data.whitelist_card_ids.value || [])]
  detailCard.value = null
  displayCount.value = BATCH_SIZE
  assetAutoTriggered.value = false
  dialog.value = true
  nextTick(() => setupObserver())
}

function confirmSelection() {
  props.data.whitelist_card_ids.value = [...localSelection.value]
  dialog.value = false
}

function cancelDialog() {
  dialog.value = false
}

function removeSelected(cardId) {
  const ids = props.data.whitelist_card_ids.value || []
  const idx = ids.indexOf(cardId)
  if (idx >= 0) {
    ids.splice(idx, 1)
    props.data.whitelist_card_ids.value = [...ids]
  }
}

function clearAll() {
  localSelection.value = []
}

// Game asset download status (for bulk downloads triggered from settings)
const assetStatus = ref({ available: false, downloadedCount: 0, downloading: false })
const assetDownloading = ref(false)
const assetAutoTriggered = ref(false)
const assetImageVersion = ref(0)
let downloadPolling = null

// Per-card on-demand full image download
const downloadingFullImageFor = ref(null)  // card_id currently being downloaded

function refreshAssetStatus(forceFresh = false) {
  apis.get_game_asset_status(forceFresh).then(res => {
    assetStatus.value = res.data
    assetDownloading.value = res.data.downloading
  }).catch(() => {})
}

function triggerSingleCardFullImageDownload(card) {
  if (downloadingFullImageFor.value === card.id) return
  downloadingFullImageFor.value = card.id
  apis.download_single_card_full(card.id).catch(() => {})
  // 轮询直到全尺寸图文件出现
  const maxAttempts = 20
  let attempts = 0
  const poll = () => {
    attempts++
    const url = `/api/game_assets/support_cards_full/${card.id}.png`
    fetch(url, { method: 'HEAD' }).then(res => {
      if (res.ok) {
        // 下载完成：更新卡片标志使大图立即显示
        card.gameAssetFullImage = true
        const inList = cards.value.find(c => c.id === card.id)
        if (inList) inList.gameAssetFullImage = true
        if (downloadingFullImageFor.value === card.id) {
          downloadingFullImageFor.value = null
        }
      } else if (attempts < maxAttempts) {
        setTimeout(poll, 2000)
      } else {
        if (downloadingFullImageFor.value === card.id) {
          downloadingFullImageFor.value = null
        }
      }
    }).catch(() => {
      if (attempts < maxAttempts) setTimeout(poll, 3000)
      else if (downloadingFullImageFor.value === card.id) downloadingFullImageFor.value = null
    })
  }
  setTimeout(poll, 2000)
}

function startPolling() {
  if (downloadPolling) return
  assetDownloading.value = true
  downloadPolling = setInterval(() => {
    apis.get_game_asset_status(true).then(res => {
      assetStatus.value = res.data
      if (!res.data.downloading) {
        assetDownloading.value = false
        clearInterval(downloadPolling)
        downloadPolling = null
        // 刷新卡片数据（图片标志已更新）
        apis.get_all_support_card(true).then(r => {
          cards.value = r.data
          assetImageVersion.value++
          if (detailCard.value) {
            const refreshed = r.data.find(c => c.id === detailCard.value.id)
            if (refreshed) detailCard.value = refreshed
          }
        }).catch(() => {})
      }
    }).catch(() => {
      clearInterval(downloadPolling)
      downloadPolling = null
      assetDownloading.value = false
    })
  }, 2000)
}

function triggerDownload() {
  assetDownloading.value = true
  apis.download_support_card_assets().then(() => {
    startPolling()
  }).catch(() => {
    assetDownloading.value = false
  })
}

const assetDownloadEnabled = computed(() => store.config.base?.enable_game_asset_download?.value ?? false)
const preferGameAsset = computed(() => store.config.base?.prefer_game_asset_image?.value ?? false)
const hasMissingGameAsset = computed(() => cards.value.some(card => !card.gameAssetImage))

function ensureAutoDownload() {
  if (!dialog.value) return
  if (assetAutoTriggered.value || assetDownloading.value) return
  if (!assetDownloadEnabled.value || !assetStatus.value.available) return
  if (!hasMissingGameAsset.value) return
  assetAutoTriggered.value = true
  triggerDownload()
}

function cardImageSrc(card) {
  if (preferGameAsset.value) {
    if (card.gameAssetImage) return `/api/game_assets/support_cards/${card.id}.png?v=${assetImageVersion.value}`
    if (card.image) return `/api/clip_image/support_cards/${card.id}.png?v=${assetImageVersion.value}`
  } else {
    if (card.image) return `/api/clip_image/support_cards/${card.id}.png?v=${assetImageVersion.value}`
    if (card.gameAssetImage) return `/api/game_assets/support_cards/${card.id}.png?v=${assetImageVersion.value}`
  }
  return null
}

function cardFullImageSrc(card) {
  if (card.gameAssetFullImage) return `/api/game_assets/support_cards_full/${card.id}.png?v=${assetImageVersion.value}`
  return cardImageSrc(card)
}

onBeforeUnmount(() => {
  if (observer) observer.disconnect()
  if (downloadPolling) {
    clearInterval(downloadPolling)
    downloadPolling = null
  }
})

watch(
  [
    dialog,
    hasMissingGameAsset,
    assetDownloadEnabled,
    () => assetStatus.value.available,
    assetDownloading,
  ],
  () => {
    ensureAutoDownload()
  },
  { immediate: true },
)

// Load cards
apis.get_all_support_card().then(res => {
  cards.value = res.data
  loading.value = false
}).catch(() => {
  loading.value = false
})
refreshAssetStatus()
</script>

<template>
  <div class="whitelist-wrapper">
    <!-- Selected cards display -->
    <div class="selected-chips" v-if="selectedCards.length > 0">
      <v-chip
        v-for="card in selectedCards"
        :key="card.id"
        size="small"
        :color="rarityColor[card.rarity]"
        closable
        @click:close="removeSelected(card.id)"
        class="selected-chip"
      >
        {{ displayName(card) }}
      </v-chip>
    </div>
    <div v-else class="no-selection-hint">
      {{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.no_selection') }}
    </div>

    <!-- Open dialog button -->
    <v-btn
      variant="tonal"
      color="primary"
      prepend-icon="md:add"
      @click="openDialog"
      :loading="loading"
    >
      {{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.open_dialog') }}
    </v-btn>

    <!-- Card selection modal -->
    <v-dialog
      v-model="dialog"
      max-width="1200"
      width="calc(100vw - 32px)"
      scrollable
    >
      <v-card class="card-dialog">
        <v-card-title class="dialog-title">
          <span>{{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.dialog_title') }}</span>
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
            :label="translateKey('backend.config.task__auto_enhancement_support_card.whitelist.search_placeholder')"
            density="compact"
            variant="outlined"
            clearable
            hide-details
            class="mb-3"
          />

          <!-- Filters -->
          <div class="filter-section">
            <div class="filter-group">
              <span class="filter-label">{{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.filter.rarity') }}</span>
              <v-chip-group v-model="filterRarity" multiple>
                <v-chip
                  v-for="opt in rarityOptions"
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
            <div class="filter-group">
              <span class="filter-label">{{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.filter.type') }}</span>
              <v-chip-group v-model="filterType" multiple>
                <v-chip
                  v-for="opt in typeOptions"
                  :key="opt.value"
                  :value="opt.value"
                  variant="outlined"
                  filter
                  size="small"
                >
                  {{ opt.label }}
                </v-chip>
              </v-chip-group>
            </div>
            <div class="filter-group">
              <span class="filter-label">{{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.filter.plan') }}</span>
              <v-chip-group v-model="filterPlan" multiple>
                <v-chip
                  v-for="opt in planOptions"
                  :key="opt.value"
                  :value="opt.value"
                  variant="outlined"
                  filter
                  size="small"
                >
                  {{ opt.label }}
                </v-chip>
              </v-chip-group>
            </div>
          </div>

          <!-- Download status bar (only when actively downloading) -->
          <div v-if="assetDownloading" class="asset-download-bar">
            <v-progress-linear
              :model-value="assetStatus.total > 0 ? (assetStatus.progress / assetStatus.total * 100) : 0"
              :indeterminate="assetStatus.total === 0"
              color="primary"
              rounded
              height="4"
              class="flex-grow-1"
            />
            <span class="asset-status-text">{{ translateAny(assetStatus.message) || translateKey('backend.config.task__auto_enhancement_support_card.whitelist.downloading_images') }}</span>
          </div>

          <!-- Main + Sidebar layout -->
          <div class="content-layout">
            <!-- Card grid (main) -->
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
                      <span class="placeholder-rarity" :style="{ color: rarityColor[card.rarity] }">
                  {{ rarityLabels[card.rarity] }}
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
                    <div class="card-rarity-badge" :style="{ backgroundColor: rarityColor[card.rarity] }">
                    {{ rarityLabels[card.rarity] }}
                    </div>
                    <!-- Info button overlay -->
                    <v-btn
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

            <!-- Detail sidebar -->
            <div class="content-sidebar" v-if="detailCard">
              <div class="detail-panel">
                <div class="detail-header">
                  <v-btn
                    icon
                    size="x-small"
                    variant="text"
                    class="detail-close"
                    @click="detailCard = null"
                  >
                    <v-icon size="18">md:close</v-icon>
                  </v-btn>
                </div>

                <!-- Full card image with rarity/level overlay -->
                <div class="detail-image-wrapper">
                  <v-img
                    v-if="cardFullImageSrc(detailCard)"
                    :src="cardFullImageSrc(detailCard)"
                    aspect-ratio="2"
                    contain
                    class="detail-image"
                  />
                  <div v-else class="card-image-placeholder detail-placeholder">
                    <span class="placeholder-rarity" :style="{ color: rarityColor[detailCard.rarity] }">
                      {{ rarityLabels[detailCard.rarity] }}
                    </span>
                  </div>
                  <!-- Rarity + Level badge -->
                  <div class="detail-badge" :style="{ backgroundColor: rarityColor[detailCard.rarity] }">
                    {{ rarityLabels[detailCard.rarity] }} Lv{{ detailLevel }}
                  </div>
                  <!-- Downloading overlay -->
                  <div v-if="downloadingFullImageFor === detailCard.id" class="detail-image-loading">
                    <v-progress-circular indeterminate size="24" width="2" color="white" />
                  </div>
                </div>

                <!-- Card name -->
                <div class="detail-name">{{ displayName(detailCard) }}</div>
                <div class="detail-name-sub" v-if="detailCard.translation?.name && detailCard.name !== detailCard.translation.name">
                  {{ detailCard.name }}
                </div>

                <!-- Tags row -->
                <div class="detail-tags">
                  <v-chip
                    size="x-small"
                    :color="rarityColor[detailCard.rarity]"
                    variant="flat"
                    class="detail-tag"
                  >
                    {{ rarityLabels[detailCard.rarity] }}
                  </v-chip>
                  <v-chip size="x-small" variant="outlined" class="detail-tag">
                    {{ typeLabels[detailCard.type] || detailCard.type }}
                  </v-chip>
                  <v-chip size="x-small" variant="outlined" class="detail-tag">
                    {{ planLabels[detailCard.planType] || detailCard.planType }}
                  </v-chip>
                  <v-chip v-if="detailCard.isLimited" size="x-small" color="red" variant="outlined" class="detail-tag">
                    {{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.limit_only') }}
                  </v-chip>
                </div>

                <!-- Level slider -->
                <div class="detail-level-slider">
                  <span class="detail-level-label">{{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.levelShort') }}</span>
                  <v-slider
                    v-model="detailLevel"
                    :min="1"
                    :max="maxLevel(detailCard)"
                    :step="1"
                    density="compact"
                    hide-details
                    color="primary"
                    thumb-label
                    class="detail-slider"
                  />
                  <span class="detail-level-value">{{ detailLevel }}</span>
                </div>

                <!-- 支援発生率 -->
                <div class="detail-trigger-rate" v-if="detailCard.produceCardUpgradePermil">
                  <span class="trigger-rate-icon">⬆</span>
                  <span class="trigger-rate-text">
                    {{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.trigger_rate') }}:
                    <strong>{{ supportTriggerRate(detailCard) }}%</strong>
                  </span>
                </div>

                <!-- サポートアビリティ (level-aware) -->
                <div class="detail-section" v-if="skillDescriptionsAtLevel(detailCard, detailLevel).length > 0">
                  <div class="detail-section-title">{{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.section.support_ability') }}</div>
                  <div class="detail-skill-list">
                    <div
                      v-for="(desc, idx) in skillDescriptionsAtLevel(detailCard, detailLevel)"
                      :key="idx"
                      class="detail-skill-item"
                    >
                      <span class="skill-indicator" :style="{ backgroundColor: rarityColor[detailCard.rarity] || '#66bb6a' }"></span>
                      <span class="skill-text">{{ desc }}</span>
                    </div>
                  </div>
                </div>

                <!-- サポートイベント (level-aware) -->
                <div class="detail-section" v-if="eventsAtLevel(detailCard, detailLevel).length > 0">
                  <div class="detail-section-title">{{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.section.support_event') }}</div>
                  <div class="detail-skill-list">
                    <div
                      v-for="evt in eventsAtLevel(detailCard, detailLevel)"
                      :key="evt.number"
                      class="detail-skill-item detail-event-entry"
                    >
                      <span class="skill-indicator" style="background-color: #42a5f5;"></span>
                      <span class="skill-text">
                        <span class="event-title-row">
                          <span class="event-title">{{ evt.title }}</span>
                          <span class="event-unlock-badge" v-if="evt.supportCardLevel > 1">Lv{{ evt.supportCardLevel }}~</span>
                        </span>
                        <span class="event-desc" v-if="evt.descriptions.length > 0">{{ evt.descriptions[0] }}</span>
                      </span>
                    </div>
                  </div>
                </div>

                <!-- P-item / スキルカード list -->
                <div class="detail-section" v-if="detailCard.eventItems && detailCard.eventItems.length > 0">
                  <div class="detail-section-title">{{ translateKey('backend.config.task__auto_enhancement_support_card.whitelist.section.attachments') }}</div>
                  <div class="detail-skill-list">
                    <div
                      v-for="item in detailCard.eventItems"
                      :key="item.id"
                      class="detail-skill-item detail-reward-item"
                    >
                      <img
                        v-if="eventItemImageSrc(item)"
                        :src="eventItemImageSrc(item)"
                        class="event-item-thumb"
                        @error="$event.target.style.display='none'"
                      />
                      <span
                        v-else
                        class="skill-indicator"
                        :style="{ backgroundColor: eventItemKindColor(item) }"
                      ></span>
                      <span class="skill-text">
                        <span class="reward-header">
                          <span class="event-item-rarity" :style="{ color: itemRarityColor(item.rarity) }">
                            {{ itemRarityLabel(item.rarity) }}
                          </span>
                          <span class="reward-name">{{ item.name }}</span>
                          <span class="reward-kind-badge" :style="{ color: eventItemKindColor(item) }">
                            {{ eventItemKindLabel(item) }}
                          </span>
                        </span>
                        <span class="reward-desc" v-if="item.descriptions && item.descriptions.length > 0">
                          <span v-for="(d, di) in item.descriptions" :key="di" class="reward-desc-line">{{ d }}</span>
                        </span>
                      </span>
                    </div>
                  </div>
                </div>

                <!-- Action button -->
                <div class="detail-actions">
                  <v-btn
                    size="small"
                    :variant="isSelected(detailCard) ? 'flat' : 'outlined'"
                    :color="isSelected(detailCard) ? 'error' : 'primary'"
                    block
                    @click="toggleCard(detailCard)"
                  >
                    {{ isSelected(detailCard) ? translateKey('dialogs.selector.cancelSelect') : translateKey('backend.config.task__auto_enhancement_support_card.whitelist.add_to_whitelist') }}
                  </v-btn>
                </div>
              </div>
            </div>
          </div>
        </v-card-text>

        <v-card-actions class="dialog-actions">
          <v-btn variant="text" size="small" @click="clearAll">{{ translateKey('dialogs.selector.clearSelection') }}</v-btn>
          <v-spacer />
          <v-btn color="error" variant="text" @click="cancelDialog">{{ translateKey('common.cancel') }}</v-btn>
          <v-btn color="primary" variant="flat" @click="confirmSelection">
            {{ translateKey('dialogs.selector.confirmMulti', { count: localSelection.length }) }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<style scoped>
.whitelist-wrapper {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.selected-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.selected-chip {
  max-width: 200px;
}

.no-selection-hint {
  color: rgba(var(--v-theme-on-surface), 0.5);
  font-size: 0.875rem;
}

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

.asset-download-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0 8px;
}

.asset-status-text {
  font-size: 0.75rem;
  color: rgba(var(--v-theme-on-surface), 0.5);
  white-space: nowrap;
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

/* Main + sidebar layout */
.content-layout {
  display: flex;
  gap: 16px;
  min-height: 0;
  /* let content-main scroll, sidebar scroll independently */
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

/* Detail sidebar */
.detail-panel {
  background: rgba(var(--v-theme-on-surface), 0.03);
  border-radius: 10px;
  padding: 12px;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.08);
}

.detail-header {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 4px;
}

.detail-image-wrapper {
  position: relative;
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 10px;
  background: rgba(var(--v-theme-on-surface), 0.05);
}

.detail-image-loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.35);
  border-radius: 8px;
}

.detail-image {
  width: 100%;
  height: 100%;
}

.detail-placeholder {
  border-radius: 8px;
  aspect-ratio: 2;
}

.detail-badge {
  position: absolute;
  bottom: 6px;
  left: 6px;
  font-size: 0.7rem;
  font-weight: 700;
  color: #000;
  padding: 2px 8px;
  border-radius: 4px;
  line-height: 1.4;
  letter-spacing: 0.02em;
}

.detail-name {
  font-size: 0.875rem;
  font-weight: 600;
  line-height: 1.3;
  margin-bottom: 2px;
}

.detail-name-sub {
  font-size: 0.75rem;
  color: rgba(var(--v-theme-on-surface), 0.45);
  margin-bottom: 8px;
  line-height: 1.3;
}

.detail-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 10px;
}

.detail-tag {
  font-size: 0.65rem;
}

.detail-section {
  margin-bottom: 10px;
}

.detail-section-title {
  font-size: 0.75rem;
  font-weight: 600;
  color: rgba(var(--v-theme-on-surface), 0.7);
  margin-bottom: 4px;
}

.detail-level-slider {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
}

.detail-level-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: rgba(var(--v-theme-on-surface), 0.6);
  flex-shrink: 0;
}

.detail-slider {
  flex: 1;
}

.detail-level-value {
  font-size: 0.75rem;
  font-weight: 700;
  color: rgba(var(--v-theme-on-surface), 0.85);
  min-width: 20px;
  text-align: right;
  flex-shrink: 0;
}

.detail-trigger-rate {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  margin-bottom: 10px;
  background: rgba(76, 175, 80, 0.1);
  border-radius: 6px;
  border: 1px solid rgba(76, 175, 80, 0.25);
}

.trigger-rate-icon {
  font-size: 0.8rem;
  color: #66bb6a;
}

.trigger-rate-text {
  font-size: 0.78rem;
  color: rgba(var(--v-theme-on-surface), 0.75);
}

.trigger-rate-text strong {
  color: #66bb6a;
  font-weight: 700;
}

.detail-skill-list {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.detail-skill-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 5px 8px;
  background: rgba(var(--v-theme-on-surface), 0.04);
  border-radius: 4px;
}

.skill-indicator {
  width: 4px;
  min-height: 14px;
  align-self: stretch;
  border-radius: 2px;
  flex-shrink: 0;
  margin-top: 1px;
}

.skill-text {
  font-size: 0.72rem;
  line-height: 1.45;
  color: rgba(var(--v-theme-on-surface), 0.7);
}

.event-item-rarity {
  font-size: 0.65rem;
  font-weight: 700;
  margin-right: 4px;
}

.detail-reward-item .skill-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.reward-header {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
}

.reward-name {
  font-weight: 600;
  color: rgba(var(--v-theme-on-surface), 0.85);
  font-size: 0.72rem;
}

.reward-kind-badge {
  font-size: 0.6rem;
  font-weight: 600;
  opacity: 0.8;
}

.reward-desc {
  font-size: 0.66rem;
  color: rgba(var(--v-theme-on-surface), 0.5);
  line-height: 1.35;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.reward-desc-line {
  display: block;
}

.event-item-thumb {
  width: 32px;
  height: 32px;
  border-radius: 4px;
  object-fit: cover;
  flex-shrink: 0;
  background: rgba(var(--v-theme-on-surface), 0.05);
}

.detail-event-entry .skill-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.event-title-row {
  display: flex;
  align-items: center;
  gap: 5px;
  flex-wrap: wrap;
}

.event-title {
  font-weight: 600;
  color: rgba(var(--v-theme-on-surface), 0.88);
  font-size: 0.72rem;
}

.event-unlock-badge {
  font-size: 0.6rem;
  font-weight: 700;
  color: #42a5f5;
  background: rgba(66, 165, 245, 0.12);
  border: 1px solid rgba(66, 165, 245, 0.35);
  border-radius: 3px;
  padding: 0px 4px;
  line-height: 1.5;
  flex-shrink: 0;
}

.event-desc {
  font-size: 0.68rem;
  color: rgba(var(--v-theme-on-surface), 0.55);
}

.detail-actions {
  margin-top: 10px;
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
    from { transform: translateY(100%); }
    to { transform: translateY(0); }
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
