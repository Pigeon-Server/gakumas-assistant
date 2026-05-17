<script setup lang="ts">
import { computed } from 'vue'

import TaskConfigField from '@/components/lists/config/task_config_field.vue'
import { translateKey } from '@/scripts/i18n/translate'

const props = defineProps({
  config: {
    type: Object,
    required: true,
  },
  section: {
    type: Object,
    default: null,
  },
  sectionName: {
    type: String,
    required: true,
  },
  save: {
    type: Function,
    default: null,
  },
})

const sectionReady = computed(() => Boolean(props.section))

const fieldEntries = computed(() => {
  if (!props.section) {
    return []
  }

  return Object.entries(props.section)
    .filter(([, item]) => item?.ui?.auto_generate !== false)
    .filter(([, item]) => isVisible(item?.ui?.visible_if))
    .map(([fieldName, item]) => ({
      fieldName,
      item,
      order: item?.ui?.order ?? 0,
    }))
    .sort((left, right) => left.order - right.order)
})

function getConfigValue(path: string) {
  const [sectionName, fieldName] = path.split('.')
  return props.config?.[sectionName]?.[fieldName]?.value
}

function isVisible(visibleIf: Record<string, any> | undefined) {
  if (!visibleIf) {
    return true
  }
  if (Array.isArray((visibleIf as Record<string, any>)?.__or__)) {
    return (visibleIf as Record<string, any>).__or__.some(rule => isVisible(rule))
  }
  if (Array.isArray((visibleIf as Record<string, any>)?.__and__)) {
    return (visibleIf as Record<string, any>).__and__.every(rule => isVisible(rule))
  }
  return Object.entries(visibleIf).every(([path, expected]) => {
    const currentValue = getConfigValue(path)
    if (Array.isArray(expected)) {
      return expected.includes(currentValue)
    }
    return currentValue === expected
  })
}
</script>

<template>
  <v-form
    v-if="sectionReady"
    v-auto-save="save"
  >
    <v-row dense class="task-config-section-form__row">
      <v-col
        v-for="entry in fieldEntries"
        :key="`${sectionName}.${entry.fieldName}`"
        cols="12"
      >
        <TaskConfigField
          :config="config"
          :section="section"
          :section-name="sectionName"
          :field-name="entry.fieldName"
          :item="entry.item"
          :save="save"
        />
      </v-col>
    </v-row>
  </v-form>
  <div v-else class="pa-4 text-body-2 text-medium-emphasis">
    {{ translateKey('common.loading') }}…{{ translateKey('config.loadingHint') }}
  </div>
</template>

<style scoped>
.task-config-section-form__row {
  padding-top: 15px;
}
</style>
