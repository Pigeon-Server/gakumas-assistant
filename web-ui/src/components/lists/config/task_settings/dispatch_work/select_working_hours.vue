<script setup>
import { computed } from "vue";

import app from "@/main.js";
import { translateAny, translateOptionTitle } from '@/scripts/i18n/translate'
const props = defineProps({
  data: Object
})

const items = computed(() => props.data?.working_hours?.ui?.options || [])
</script>

<template>
  <v-select
    :label="translateAny(data.working_hours.ui.label)"
    :hint="translateAny(data.working_hours.ui.hint)"
    :items="items"
    :item-title="translateOptionTitle"
    item-value="value"
    :item-color="app.config.globalProperties.$theme.color"
    v-model="data.working_hours.value"
    persistent-hint
  >
    <template #item="{ props: optionProps, item: optionItem }">
      <v-list-item v-bind="optionProps" :title="translateAny(optionItem.raw?.title)" />
    </template>
    <template #selection="{ item: optionItem }">
      <span>{{ translateAny(optionItem.raw?.title) }}</span>
    </template>
  </v-select>
</template>

<style scoped>

</style>
