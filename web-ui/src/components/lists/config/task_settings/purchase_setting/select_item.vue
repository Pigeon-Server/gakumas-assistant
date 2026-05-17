<script setup>
import apis from "@/scripts/apis.js";
import app from "@/main.js";
import { useAppStore } from "@/stores/app.ts";
import { ref } from "vue";
import { translateAny } from '@/scripts/i18n/translate'

const store = useAppStore()

const props = defineProps({
  data: Object,
  label: {
    type: [String, Object],
    default: ""
  }
})

const items = ref([])
apis.get_all_item().then(res => {
  items.value = res.data
})

function filter(value, queryText, item) {
  const toLowerCaseString = val =>
    String(val != null ? val : '').toLowerCase()
  const query = toLowerCaseString(queryText)
  return item.raw.name.includes(query) || item.raw.translation?.name.includes(query)
}

function itemImageSrc(item) {
  if (item.gameAssetImage) return `/api/game_assets/items/${item.id}.png`
  if (item.image) return `/api/clip_image/items/${item.id}.png`
  return null
}
</script>

<template>
  <v-select
    v-model="data.daily_buy_list.value"
    :items="items"
    item-value="id"
    :label="translateAny(label) || translateAny(data.daily_buy_list.ui.label)"
    multiple
    clearable
    :custom-filter="filter"
  >
    <template v-slot:item="{ item, props }">
      <v-list-item
        v-bind="props"
        :title="item.raw.translation?.name || item.raw.name"
        :subtitle="item.raw.translation?.name ? item.raw.name : ''"
        :key="item.raw.id" class="select_item">
        <template v-slot:prepend>
          <v-img
            v-if="itemImageSrc(item.raw)"
            :src="itemImageSrc(item.raw)"
            width="48"
            height="48"
          />
        </template>
        <div class="item-content">
          <div class="item-description">
            {{ item.raw.translation?.description || item.raw.description }}
          </div>

          <div class="item-route">
            {{ item.raw.translation?.acquisitionRouteDescription || item.raw.acquisitionRouteDescription }}
          </div>
        </div>
      </v-list-item>
    </template>

    <template v-slot:selection="{ item, index }">
      <v-chip
        :text="item.raw.translation?.name || item.raw.name"
        size="small"
        :color="app.config.globalProperties.$theme.color"
      >
        <template v-slot:prepend>
          <v-img
            v-if="itemImageSrc(item.raw)"
            :src="itemImageSrc(item.raw)"
            width="24"
            height="24"
            class="mr-2"
          />
        </template>
      </v-chip>
    </template>
  </v-select>
</template>

<style scoped>
.select_item {
  width: 100%;
  max-width: min(400px, calc(100vw - 96px));
  box-sizing: border-box;
  display: flex;
  align-items: flex-start;
  padding: 8px 12px;
  gap: 12px;
}

.item-image {
  flex: 0 0 auto;
}

.item-content {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.item-title {
  font-size: 15px;
  font-weight: 600;
  color: rgba(var(--v-theme-on-surface), 0.95);
}

.item-subtitle {
  font-size: 13px;
  color: rgba(var(--v-theme-on-surface), 0.65);
  margin-left: 4px;
}

.item-description {
  font-size: 13px;
  color: rgba(var(--v-theme-on-surface), 0.75);
  line-height: 1.3;
  word-break: break-word;
}

.item-route {
  font-size: 12px;
  color: rgba(var(--v-theme-on-surface), 0.55);
  line-height: 1.2;
  word-break: break-word;
}

@media (max-width: 599px) {
  .select_item {
    max-width: calc(100vw - 72px);
    padding: 8px 10px;
    gap: 10px;
  }
}

</style>
