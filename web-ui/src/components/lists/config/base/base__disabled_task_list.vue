<script setup>
import { computed } from "vue";
import app from "@/main.js";
import {useAppStore} from "@/stores/app.ts";
import { translateAny } from '@/scripts/i18n/translate'

const store = useAppStore();
const taskOptions = computed(() => (
  Object.entries(store.task_list).map(([taskName, task]) => ({
    value: taskName,
    title: translateAny(task.description),
  }))
));
</script>

<template>
  <v-list-item>
    <v-autocomplete
      v-model="store.config.base.disabled_tasks.value"
      :items="taskOptions"
      item-title="title"
      item-value="value"
      :item-color="app.config.globalProperties.$theme.color"
      :label="translateAny(store.config.base.disabled_tasks.ui.label)"
      :hint="translateAny(store.config.base.disabled_tasks.ui.hint)"
      persistent-hint
      chips
      multiple
    >
      <template #item="{ props, item }">
        <v-list-item
          v-bind="props"
          :title="item.raw.title"
        />
      </template>
      <template #selection="{ item }">
        <span>{{ item.raw.title }}</span>
      </template>
    </v-autocomplete>
  </v-list-item>
</template>

<style scoped>

</style>
