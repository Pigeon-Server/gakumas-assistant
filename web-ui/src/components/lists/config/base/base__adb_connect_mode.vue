<script setup>
import app from "@/main.js";
import { translateAny, translateOptionTitle } from '@/scripts/i18n/translate'
const props = defineProps({
  data: Object
})
</script>

<template>
  <v-list-item>
    <v-select
      :label="translateAny(props.data.base.adb_connect_mode.ui.label)"
      :hint="translateAny(props.data.base.adb_connect_mode.ui.hint)"
      :items="props.data.base.adb_connect_mode.ui.options"
      :item-title="translateOptionTitle"
      item-value="value"
      :item-color="app.config.globalProperties.$theme.color"
      v-model="props.data.base.adb_connect_mode.value"
      persistent-hint
    >
      <template #item="{ props: optionProps, item: optionItem }">
        <v-list-item v-bind="optionProps" :title="translateAny(optionItem.raw?.title)" />
      </template>
      <template #selection="{ item: optionItem }">
        <span>{{ translateAny(optionItem.raw?.title) }}</span>
      </template>
    </v-select>
  </v-list-item>
</template>

<style scoped>

</style>
