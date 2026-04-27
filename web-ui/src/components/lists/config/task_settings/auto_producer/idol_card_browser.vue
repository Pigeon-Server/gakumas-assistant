<script setup>
  import { computed, onBeforeUnmount, ref, watch } from 'vue'
  import CardSelectorDialog from '@/components/dialogs/CardSelectorDialog.vue'
  import apis from '@/scripts/apis.js'
  import { probeImage } from '@/scripts/utils/image.js'
  import { useAppStore } from '@/stores/app.ts'

  const store = useAppStore()

  const props = defineProps({
    data: Object,
    taskName: {
      type: String,
      default: 'auto_producer',
    },
    save: {
      type: Function,
      default: null,
    },
  })

  const dialog = ref(false)
  const idolCards = ref([])
  const idolCardsLoading = ref(true)
  const downloadingFullImageFor = ref(null)
  const assetStatus = ref({ available: false, downloadedCount: 0, downloading: false })
  const assetDownloading = ref(false)
  const assetAutoTriggered = ref(false)
  const idolCardSkin = ref(0)
  const assetImageVersion = ref(0)
  const imageRetryKey = ref(0)

  const managedTimers = new Set()
  let imageRetryTimer = null
  let downloadPolling = null

  function startManagedTimer (callback, delay) {
    const timer = window.setTimeout(() => {
      managedTimers.delete(timer)
      callback()
    }, delay)
    managedTimers.add(timer)
    return timer
  }

  function clearManagedTimer (timer) {
    if (timer === null) return
    window.clearTimeout(timer)
    managedTimers.delete(timer)
  }

  const idolCardField = computed(() =>
    props.data?.target_idol_card_id ?? props.data?.target_idol_card_name ?? null,
  )
  const assetDownloadEnabled = computed(() => store.config.base?.enable_game_asset_download?.value ?? false)
  const preferGameAsset = computed(() => store.config.base?.prefer_game_asset_image?.value ?? false)
  const hasMissingGameAsset = computed(() => idolCards.value.some(card => !card.gameAssetImage))

  apis.get_all_idol_card().then(res => {
    idolCards.value = (res.data || []).map(card => ({
      ...card,
      _hasFullImage: !!card.hasFullImage,
      _fullImageVersion: 0,
    }))
    idolCardsLoading.value = false
  }).catch(() => {
    idolCardsLoading.value = false
  })

  const idolRarityColor = {
    IdolCardRarity_Ssr: '#FFD700',
    IdolCardRarity_Sr: '#C0C0C0',
    IdolCardRarity_R: '#CD7F32',
  }

  const idolRarityLabel = {
    IdolCardRarity_Ssr: 'SSR',
    IdolCardRarity_Sr: 'SR',
    IdolCardRarity_R: 'R',
  }

  const idolPlanLabel = {
    ProducePlanType_Plan1: '感性',
    ProducePlanType_Plan2: '逻辑',
    ProducePlanType_Plan3: '异常',
    ProducePlanType_Common: '通用',
  }

  const examEffectLabel = {
    ProduceExamEffectType_ExamParameterBuff: '好調',
    ProduceExamEffectType_ExamReview: '好印象',
    ProduceExamEffectType_ExamLessonBuff: '集中',
    ProduceExamEffectType_ExamConcentration: '強気',
    ProduceExamEffectType_ExamCardPlayAggressive: 'やる気',
    ProduceExamEffectType_ExamFullPower: '全力',
  }

  const characterOptions = computed(() => {
    const characterMap = new Map()
    for (const card of idolCards.value) {
      if (card.characterId && card.characterName && !characterMap.has(card.characterId)) {
        characterMap.set(card.characterId, card.characterName)
      }
    }

    return Array.from(characterMap.entries())
      .toSorted((left, right) => left[1].localeCompare(right[1], 'ja'))
      .map(([value, label]) => ({ value, label }))
  })

  const idolFilterGroups = computed(() => [
    {
      label: '稀有度',
      key: 'rarity',
      options: [
        { value: 'IdolCardRarity_Ssr', label: 'SSR', color: '#FFD700' },
        { value: 'IdolCardRarity_Sr', label: 'SR', color: '#C0C0C0' },
        { value: 'IdolCardRarity_R', label: 'R', color: '#CD7F32' },
      ],
    },
    {
      label: '计划',
      key: 'planType',
      options: [
        { value: 'ProducePlanType_Plan1', label: '感性' },
        { value: 'ProducePlanType_Plan2', label: '逻辑' },
        { value: 'ProducePlanType_Plan3', label: '异常' },
      ],
    },
    {
      label: '属性',
      key: 'primaryAttribute',
      options: [
        { value: 'vocal', label: '声乐 Vo.' },
        { value: 'dance', label: '舞蹈 Da.' },
        { value: 'visual', label: '形象 Vi.' },
      ],
    },
    {
      label: '角色流派',
      key: 'examEffectType',
      options: [
        { value: 'ProduceExamEffectType_ExamParameterBuff', label: '好調' },
        { value: 'ProduceExamEffectType_ExamReview', label: '好印象' },
        { value: 'ProduceExamEffectType_ExamLessonBuff', label: '集中' },
        { value: 'ProduceExamEffectType_ExamConcentration', label: '強気' },
        { value: 'ProduceExamEffectType_ExamFullPower', label: '全力' },
        { value: 'ProduceExamEffectType_ExamCardPlayAggressive', label: 'やる気' },
      ],
    },
    {
      label: '角色',
      key: 'characterId',
      options: characterOptions.value,
    },
  ])

  function idolCardImageSrc (card) {
    if (preferGameAsset.value) {
      if (card?.gameAssetImage) return `/api/game_assets/idol_cards/${card.id}.png?v=${assetImageVersion.value}`
      if (card?.hasImage) return `/api/clip_image/idol_cards/${card.id}.png?v=${assetImageVersion.value}`
    } else {
      if (card?.hasImage) return `/api/clip_image/idol_cards/${card.id}.png?v=${assetImageVersion.value}`
      if (card?.gameAssetImage) return `/api/game_assets/idol_cards/${card.id}.png?v=${assetImageVersion.value}`
    }
    return null
  }

  function idolCardFullImageTriggerSrc (card, skin = idolCardSkin.value) {
    const version = card?._fullImageVersion || 0
    return `/api/game_assets/idol_cards_full/${card.id}_${skin}.png?v=${version}`
  }

  function idolCardFullImageSrc (card, skin = idolCardSkin.value) {
    const version = card?._fullImageVersion || 0
    return `/api/game_assets/idol_cards_full/${card.id}_${skin}.png?v=${version}`
  }

  function idolCardDetailImageSrc (card) {
    if (card?._hasFullImage) {
      return idolCardFullImageSrc(card)
    }
    return idolCardImageSrc(card)
  }

  function localizedName (item) {
    return item?.translation?.name || item?.name || ''
  }

  function localizedDesc (item) {
    return item?.translation?.description || item?.description || ''
  }

  function skillCardImageSrc (card) {
    if (!card?.assetId || !card?.hasImage) return null
    const stripped = card.assetId.replace('img_general_', '')
    return `/api/game_assets/skill_cards/${stripped}.png?v=${imageRetryKey.value}`
  }

  function produceItemImageSrc (item) {
    if (!item?.assetId || !item?.hasImage) return null
    const stripped = item.assetId.replace('img_general_', '')
    return `/api/game_assets/items/${stripped}.png?v=${imageRetryKey.value}`
  }

  function idolCardDisplayName (card) {
    return card?.translation?.name || card?.name || ''
  }

  function idolCardSubtitle (card) {
    const rarity = idolRarityLabel[card.rarity] || ''
    const plan = idolPlanLabel[card.planType] || ''
    const character = card.characterName || ''
    return [rarity, plan, character].filter(Boolean).join(' · ')
  }

  function getIdolRarityColor (card) {
    return idolRarityColor[card.rarity] || '#888'
  }

  function getIdolRarityLabel (card) {
    return idolRarityLabel[card.rarity] || ''
  }

  function formatGrowthRate (permil) {
    if (!permil) return ''
    return `${(permil / 10).toFixed(1)}%`
  }

  function scheduleImageRetry () {
    if (imageRetryTimer !== null) return
    imageRetryTimer = startManagedTimer(() => {
      imageRetryKey.value++
      imageRetryTimer = null
    }, 1200)
  }

  function syncCardImageState (card) {
    card._hasFullImage = true
    card._fullImageVersion = (card._fullImageVersion || 0) + 1

    const cardInList = idolCards.value.find(item => item.id === card.id)
    if (cardInList && cardInList !== card) {
      cardInList._hasFullImage = true
      cardInList._fullImageVersion = card._fullImageVersion
    }
  }

  function refreshAssetStatus (forceFresh = false) {
    apis.get_game_asset_status(forceFresh).then(res => {
      assetStatus.value = res.data
      assetDownloading.value = res.data.downloading
    }).catch(() => {})
  }

  function startPolling () {
    if (downloadPolling) return
    assetDownloading.value = true
    downloadPolling = window.setInterval(() => {
      apis.get_game_asset_status(true).then(res => {
        assetStatus.value = res.data
        if (!res.data.downloading) {
          assetDownloading.value = false
          window.clearInterval(downloadPolling)
          downloadPolling = null
          apis.get_all_idol_card(true).then(r => {
            idolCards.value = (r.data || []).map(card => ({
              ...card,
              _hasFullImage: !!card.hasFullImage,
              _fullImageVersion: 0,
            }))
            assetImageVersion.value++
          }).catch(() => {})
        }
      }).catch(() => {
        window.clearInterval(downloadPolling)
        downloadPolling = null
        assetDownloading.value = false
      })
    }, 2000)
  }

  function triggerDownload () {
    assetDownloading.value = true
    apis.download_idol_card_assets().then(() => {
      startPolling()
    }).catch(() => {
      assetDownloading.value = false
    })
  }

  function ensureAutoDownload () {
    if (!dialog.value) return
    if (assetAutoTriggered.value || assetDownloading.value) return
    if (!assetDownloadEnabled.value || !assetStatus.value.available) return
    if (!hasMissingGameAsset.value) return
    assetAutoTriggered.value = true
    triggerDownload()
  }

  function triggerIdolCardFullImageDownload (card, skin = idolCardSkin.value) {
    const downloadKey = `${card.id}_${skin}`
    if (downloadingFullImageFor.value === downloadKey) return

    downloadingFullImageFor.value = downloadKey
    const fullImageUrl = idolCardFullImageSrc(card, skin)
    apis.download_single_idol_card_full(card.id, skin).catch(() => {})

    let attempts = 0
    const maxAttempts = 20
    const poll = () => {
      attempts++
      probeImage(fullImageUrl).then(exists => {
        if (exists) {
          syncCardImageState(card)
          if (downloadingFullImageFor.value === downloadKey) {
            downloadingFullImageFor.value = null
          }
          return
        }

        if (attempts < maxAttempts) {
          startManagedTimer(poll, 2000)
        } else if (downloadingFullImageFor.value === downloadKey) {
          downloadingFullImageFor.value = null
        }
      }).catch(() => {
        if (attempts < maxAttempts) {
          startManagedTimer(poll, 3000)
        } else if (downloadingFullImageFor.value === downloadKey) {
          downloadingFullImageFor.value = null
        }
      })
    }

    startManagedTimer(poll, 2000)
  }

  function onIdolCardDetail (card) {
    idolCardSkin.value = 0
    if (!card._hasFullImage && assetDownloadEnabled.value && assetStatus.value.available) {
      triggerIdolCardFullImageDownload(card, 0)
    }
  }

  function onIdolCardDetailImageError (card) {
    if (card?._hasFullImage) {
      card._hasFullImage = false
      scheduleImageRetry()
    }
  }

  function clearSelectedCard () {
    if (idolCardField.value) {
      idolCardField.value.value = ''
    }
  }

  const selectedIds = computed({
    get () {
      const value = idolCardField.value?.value
      return value ? [value] : []
    },
    set (ids) {
      if (idolCardField.value) {
        idolCardField.value.value = ids.length > 0 ? ids[0] : ''
      }
    },
  })

  const selectedIdolCard = computed(() => {
    const cardId = idolCardField.value?.value
    if (!cardId) return null
    return idolCards.value.find(card => card.id === cardId) || null
  })

  function openDialog () {
    assetAutoTriggered.value = false
    dialog.value = true
    ensureAutoDownload()
  }

  onBeforeUnmount(() => {
    clearManagedTimer(imageRetryTimer)
    imageRetryTimer = null
    if (downloadPolling) {
      window.clearInterval(downloadPolling)
      downloadPolling = null
    }
    for (const timer of managedTimers) {
      window.clearTimeout(timer)
    }
    managedTimers.clear()
  })

  watch(
    () => idolCardField.value?.value ?? '',
    (value, previousValue) => {
      if (value === previousValue || !props.taskName) {
        return
      }
      if (typeof props.save === 'function') {
        void props.save()
        return
      }
      void store.save_task_config(props.taskName)
    },
  )

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

  refreshAssetStatus()
</script>

<template>
  <div class="idol-card-browser">
    <div class="browser-label">目标偶像卡</div>
    <div class="browser-hint">目标 Pアイドル（留空使用默认选中的卡）</div>

    <div v-if="selectedIdolCard" class="selected-chips">
      <v-chip
        class="selected-chip"
        closable
        :color="getIdolRarityColor(selectedIdolCard)"
        size="small"
        @click:close="clearSelectedCard"
      >
        {{ idolCardDisplayName(selectedIdolCard) }}
        <template v-if="selectedIdolCard.characterName">
          · {{ selectedIdolCard.characterName }}
        </template>
      </v-chip>
    </div>
    <div v-else class="no-selection-hint">
      未选择，使用默认选中的卡
    </div>

    <v-btn
      color="primary"
      :loading="idolCardsLoading"
      prepend-icon="md:person_search"
      size="small"
      variant="tonal"
      @click="openDialog"
    >
      选择目标偶像卡
    </v-btn>

    <CardSelectorDialog
      v-model="dialog"
      v-model:selected="selectedIds"
      :card-image-src="idolCardImageSrc"
      :card-subtitle="idolCardSubtitle"
      :cards="idolCards"
      :display-name="idolCardDisplayName"
      :filter-groups="idolFilterGroups"
      :loading="idolCardsLoading"
      mode="single"
      :rarity-color="getIdolRarityColor"
      :rarity-label="getIdolRarityLabel"
      search-placeholder="搜索偶像卡名称/角色名（支持中文/日文/ID）"
      title="选择目标偶像卡"
      @card-detail="onIdolCardDetail"
    >
      <template #detail="{ card, isSelected, toggle, close }">
        <div class="detail-panel">
          <div class="detail-header">
            <v-btn
              class="detail-close"
              icon
              size="x-small"
              variant="text"
              @click="close"
            >
              <v-icon size="18">md:close</v-icon>
            </v-btn>
          </div>

          <div class="detail-image-wrapper">
            <v-img
              v-if="idolCardDetailImageSrc(card)"
              :aspect-ratio="card._hasFullImage ? 0.56 : 1"
              class="detail-image"
              contain
              :src="idolCardDetailImageSrc(card)"
              @error="onIdolCardDetailImageError(card)"
            />
            <div v-else class="card-image-placeholder detail-placeholder">
              <span class="placeholder-rarity" :style="{ color: idolRarityColor[card.rarity] }">
                {{ idolRarityLabel[card.rarity] }}
              </span>
            </div>
            <div class="detail-badge" :style="{ backgroundColor: idolRarityColor[card.rarity] }">
              {{ idolRarityLabel[card.rarity] }}
            </div>
            <div
              v-if="downloadingFullImageFor?.startsWith(card.id)"
              class="detail-image-loading"
            >
              <v-progress-circular
                color="white"
                indeterminate
                size="24"
                width="2"
              />
            </div>
          </div>

          <div v-if="card._hasFullImage" class="detail-skin-toggle">
            <v-btn-toggle v-model="idolCardSkin" color="primary" density="compact" mandatory>
              <v-btn size="x-small" :value="0">覚醒前</v-btn>
              <v-btn size="x-small" :value="1">覚醒後</v-btn>
            </v-btn-toggle>
          </div>

          <div class="detail-name">{{ idolCardDisplayName(card) }}</div>
          <div
            v-if="card.translation?.name && card.name !== card.translation.name"
            class="detail-name-sub"
          >
            {{ card.name }}
          </div>

          <div class="detail-tags">
            <v-chip class="detail-tag" :color="idolRarityColor[card.rarity]" size="x-small" variant="flat">
              {{ idolRarityLabel[card.rarity] }}
            </v-chip>
            <v-chip class="detail-tag" size="x-small" variant="outlined">
              {{ idolPlanLabel[card.planType] || card.planType }}
            </v-chip>
            <v-chip v-if="card.characterName" class="detail-tag" size="x-small" variant="outlined">
              {{ card.characterName }}
            </v-chip>
            <v-chip
              v-if="examEffectLabel[card.examEffectType]"
              class="detail-tag"
              color="teal"
              size="x-small"
              variant="outlined"
            >
              {{ examEffectLabel[card.examEffectType] }}
            </v-chip>
            <v-chip
              v-if="card.isLimited"
              class="detail-tag"
              color="red"
              size="x-small"
              variant="outlined"
            >
              限定
            </v-chip>
          </div>

          <div class="detail-section">
            <div class="detail-section-title">培育パラメータ</div>
            <div class="detail-stats">
              <div class="stat-row">
                <span class="stat-label stat-vocal">Vo</span>
                <div class="stat-bar-wrapper">
                  <div
                    class="stat-bar stat-bar-vocal"
                    :style="{ width: `${Math.min(card.produceVocal, 100)}%` }"
                  />
                </div>
                <span class="stat-value">{{ card.produceVocal }}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label stat-dance">Da</span>
                <div class="stat-bar-wrapper">
                  <div
                    class="stat-bar stat-bar-dance"
                    :style="{ width: `${Math.min(card.produceDance, 100)}%` }"
                  />
                </div>
                <span class="stat-value">{{ card.produceDance }}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label stat-visual">Vi</span>
                <div class="stat-bar-wrapper">
                  <div
                    class="stat-bar stat-bar-visual"
                    :style="{ width: `${Math.min(card.produceVisual, 100)}%` }"
                  />
                </div>
                <span class="stat-value">{{ card.produceVisual }}</span>
              </div>
            </div>
            <div class="stat-summary">
              <span class="stat-summary-item">
                总计 <strong>{{ card.produceVocal + card.produceDance + card.produceVisual }}</strong>
              </span>
              <span class="stat-summary-item stat-stamina-text">
                体力 <strong>{{ card.produceStamina }}</strong>
              </span>
            </div>
            <div class="growth-rates">
              <span class="growth-item growth-vocal">⬆{{ formatGrowthRate(card.produceVocalGrowthRatePermil) }}</span>
              <span class="growth-item growth-dance">⬆{{ formatGrowthRate(card.produceDanceGrowthRatePermil) }}</span>
              <span class="growth-item growth-visual">⬆{{ formatGrowthRate(card.produceVisualGrowthRatePermil) }}</span>
            </div>
          </div>

          <div v-if="card.produceCard" class="detail-section">
            <div class="detail-section-title">固有技能卡</div>
            <div class="detail-item-block">
              <div class="detail-item-row">
                <img
                  v-if="skillCardImageSrc(card.produceCard)"
                  class="detail-item-icon"
                  :src="skillCardImageSrc(card.produceCard)"
                  @error="scheduleImageRetry"
                >
                <div class="detail-item-text">
                  <div class="detail-item-name">{{ localizedName(card.produceCard) }}</div>
                  <div
                    v-if="card.produceCard.translation?.name && card.produceCard.translation.name !== card.produceCard.name"
                    class="detail-item-name-sub"
                  >
                    {{ card.produceCard.name }}
                  </div>
                </div>
              </div>
              <div v-if="localizedDesc(card.produceCard)" class="detail-item-desc">
                {{ localizedDesc(card.produceCard) }}
              </div>
            </div>
          </div>

          <div v-if="card.beforeProduceItem || card.afterProduceItem" class="detail-section">
            <div class="detail-section-title">固有P物品</div>

            <div v-if="card.beforeProduceItem" class="detail-item-block">
              <div class="detail-item-row">
                <img
                  v-if="produceItemImageSrc(card.beforeProduceItem)"
                  class="detail-item-icon"
                  :src="produceItemImageSrc(card.beforeProduceItem)"
                  @error="scheduleImageRetry"
                >
                <div class="detail-item-text">
                  <div class="detail-item-name">{{ localizedName(card.beforeProduceItem) }}</div>
                  <div
                    v-if="card.beforeProduceItem.translation?.name && card.beforeProduceItem.translation.name !== card.beforeProduceItem.name"
                    class="detail-item-name-sub"
                  >
                    {{ card.beforeProduceItem.name }}
                  </div>
                </div>
              </div>
              <div v-if="localizedDesc(card.beforeProduceItem)" class="detail-item-desc">
                {{ localizedDesc(card.beforeProduceItem) }}
              </div>
            </div>

            <div v-if="card.afterProduceItem" class="detail-item-block">
              <div class="detail-item-row">
                <img
                  v-if="produceItemImageSrc(card.afterProduceItem)"
                  class="detail-item-icon"
                  :src="produceItemImageSrc(card.afterProduceItem)"
                  @error="scheduleImageRetry"
                >
                <div class="detail-item-text">
                  <div class="detail-item-name">
                    {{ localizedName(card.afterProduceItem) }}
                    <v-chip class="detail-tag detail-inline-chip" color="amber" size="x-small" variant="outlined">
                      特訓後
                    </v-chip>
                  </div>
                  <div
                    v-if="card.afterProduceItem.translation?.name && card.afterProduceItem.translation.name !== card.afterProduceItem.name"
                    class="detail-item-name-sub"
                  >
                    {{ card.afterProduceItem.name }}
                  </div>
                </div>
              </div>
              <div v-if="localizedDesc(card.afterProduceItem)" class="detail-item-desc">
                {{ localizedDesc(card.afterProduceItem) }}
              </div>
            </div>
          </div>

          <div class="detail-card-id">{{ card.id }}</div>

          <div class="detail-actions">
            <v-btn
              block
              :color="isSelected ? 'error' : 'primary'"
              size="small"
              :variant="isSelected ? 'flat' : 'outlined'"
              @click="toggle"
            >
              {{ isSelected ? '取消选择' : '选择此卡' }}
            </v-btn>
          </div>
        </div>
      </template>
    </CardSelectorDialog>
  </div>
</template>

<style scoped>
.idol-card-browser {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 12px;
}

.browser-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.87);
}

.browser-hint {
  margin-top: -6px;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
}

.selected-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.selected-chip {
  max-width: 240px;
}

.no-selection-hint {
  color: rgba(255, 255, 255, 0.5);
  font-size: 0.875rem;
}

.detail-panel {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  padding: 12px;
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
  background: rgba(255, 255, 255, 0.05);
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
  aspect-ratio: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.03);
}

.placeholder-rarity {
  font-size: 1.5rem;
  font-weight: 700;
  opacity: 0.5;
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

.detail-skin-toggle {
  display: flex;
  justify-content: center;
  margin-bottom: 10px;
}

.detail-name {
  font-size: 0.875rem;
  font-weight: 600;
  line-height: 1.3;
  margin-bottom: 2px;
}

.detail-name-sub {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.45);
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

.detail-inline-chip {
  margin-left: 4px;
}

.detail-section {
  margin-bottom: 10px;
}

.detail-section-title {
  font-size: 0.75rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 4px;
}

.detail-stats {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.8rem;
}

.stat-label {
  width: 24px;
  font-weight: 700;
  font-size: 0.65rem;
  text-align: center;
  padding: 1px 4px;
  border-radius: 3px;
  color: #fff;
  flex-shrink: 0;
}

.stat-vocal {
  background: #e53935;
}

.stat-dance {
  background: #1e88e5;
}

.stat-visual {
  background: #f9a825;
}

.stat-bar-wrapper {
  flex: 1;
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
  overflow: hidden;
}

.stat-bar {
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s ease;
}

.stat-bar-vocal {
  background: #e53935;
}

.stat-bar-dance {
  background: #1e88e5;
}

.stat-bar-visual {
  background: #f9a825;
}

.stat-value {
  font-weight: 600;
  min-width: 24px;
  font-size: 0.75rem;
  text-align: right;
  flex-shrink: 0;
}

.stat-summary {
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
  color: rgba(255, 255, 255, 0.6);
  padding-top: 2px;
}

.stat-stamina-text {
  color: #66bb6a;
}

.growth-rates {
  display: flex;
  gap: 8px;
  font-size: 0.65rem;
  margin-top: 6px;
}

.growth-item {
  opacity: 0.7;
}

.growth-vocal {
  color: #ef9a9a;
}

.growth-dance {
  color: #90caf9;
}

.growth-visual {
  color: #fff176;
}

.detail-item-block {
  background: rgba(255, 255, 255, 0.04);
  border-radius: 6px;
  padding: 8px 10px;
}

.detail-item-block + .detail-item-block {
  margin-top: 8px;
}

.detail-item-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.detail-item-icon {
  width: 36px;
  height: 36px;
  border-radius: 4px;
  object-fit: cover;
  flex-shrink: 0;
  background: rgba(255, 255, 255, 0.06);
}

.detail-item-text {
  flex: 1;
  min-width: 0;
}

.detail-item-name {
  font-size: 0.8rem;
  font-weight: 600;
  margin-bottom: 2px;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.detail-item-name-sub {
  font-size: 0.65rem;
  color: rgba(255, 255, 255, 0.4);
}

.detail-item-desc {
  margin-top: 6px;
  font-size: 0.7rem;
  color: rgba(255, 255, 255, 0.55);
  line-height: 1.4;
}

.detail-card-id {
  font-size: 0.65rem;
  font-family: monospace;
  color: rgba(255, 255, 255, 0.3);
  word-break: break-all;
  text-align: center;
}

.detail-actions {
  margin-top: 10px;
}
</style>
