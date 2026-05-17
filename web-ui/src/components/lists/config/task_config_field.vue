<script setup lang="ts">
import { computed } from 'vue'

import app from '@/main.js'
import { taskCustomComponentRegistry } from '@/components/lists/config/task_custom_component_registry'
import { formatLocalizedList, translateAny, translateKey, translateOptionTitle } from '@/scripts/i18n/translate'

const props = defineProps({
  config: {
    type: Object,
    required: true,
  },
  section: {
    type: Object,
    required: true,
  },
  sectionName: {
    type: String,
    required: true,
  },
  fieldName: {
    type: String,
    required: true,
  },
  item: {
    type: Object,
    required: true,
  },
  save: {
    type: Function,
    default: null,
  },
})

const themeColor = app.config.globalProperties.$theme.color
const selectItems = computed(() => props.item?.ui?.options || [])
const componentProps = computed(() => props.item?.ui?.component_props || {})

const componentType = computed(() => {
  if (props.item?.ui?.component) {
    return props.item.ui.component
  }
  if (props.item?.data_type === 'bool') {
    return 'switch'
  }
  if (props.item?.ui?.options?.length) {
    return 'select'
  }
  return 'text'
})

const customComponentEntry = computed(() => taskCustomComponentRegistry[componentType.value] || null)
const customComponentProps = computed(() => {
  if (!customComponentEntry.value?.buildProps) {
    return {}
  }
  return customComponentEntry.value.buildProps({
    config: props.config,
    section: props.section,
    sectionName: props.sectionName,
    fieldName: props.fieldName,
    item: props.item,
    save: props.save,
  })
})

const selectHint = computed(() => {
  const baseHint = translateAny(props.item?.ui?.hint)
  const disabledMessages = selectItems.value
    .filter(option => option?.disabled && option?.disabled_reason)
    .map(option => translateKey('config.optionDisabledItem', {
      title: translateAny(option.title),
      reason: translateAny(option.disabled_reason),
    }))

  if (!disabledMessages.length) {
    return baseHint
  }

  const unavailableHint = translateKey('config.unavailableOptionsPrefix', {
    items: formatLocalizedList(disabledMessages, { style: 'long', type: 'conjunction' }),
  })
  return [baseHint, unavailableHint].filter(Boolean).join(' ')
})

function cloneValue(value: any) {
  if (value === null || value === undefined) {
    return value
  }
  return JSON.parse(JSON.stringify(value))
}

function resetValue() {
  props.item.value = cloneValue(props.item.default_value)
}

function selectItemProps(option: any) {
  return {
    disabled: Boolean(option?.disabled),
  }
}
</script>

<template>
  <component
    :is="customComponentEntry.component"
    v-if="customComponentEntry"
    v-bind="customComponentProps"
  />
  <v-switch
    v-else-if="componentType === 'switch'"
    class="task-config-field task-config-field--switch"
    v-model="item.value"
    :label="translateAny(item.ui?.label)"
    :hint="translateAny(item.ui?.hint)"
    :color="themeColor"
    density="comfortable"
    persistent-hint
    v-bind="componentProps"
  />
  <v-select
    v-else-if="componentType === 'select'"
    class="task-config-field"
    v-model="item.value"
    :items="selectItems"
    :label="translateAny(item.ui?.label)"
    :hint="selectHint"
    :color="themeColor"
    :item-color="themeColor"
    :item-props="selectItemProps"
    :item-title="translateOptionTitle"
    item-value="value"
    density="comfortable"
    persistent-hint
    v-bind="componentProps"
  >
    <template #item="{ props: optionProps, item: optionItem }">
      <v-list-item
        v-bind="optionProps"
        :title="translateAny(optionItem.raw?.title)"
        :subtitle="translateAny(optionItem.raw?.disabled_reason || optionItem.raw?.description)"
      />
    </template>
    <template #selection="{ item: optionItem }">
      <span>{{ translateAny(optionItem.raw?.title) }}</span>
    </template>
  </v-select>
  <v-text-field
    v-else-if="componentType === 'time'"
    class="task-config-field"
    v-model="item.value"
    :label="translateAny(item.ui?.label)"
    :hint="translateAny(item.ui?.hint)"
    :color="themeColor"
    prepend-inner-icon="md:schedule"
    density="comfortable"
    persistent-hint
    readonly
    v-bind="componentProps"
  >
    <v-menu
      :close-on-content-click="false"
      activator="parent"
      min-width="0"
    >
      <v-time-picker v-model="item.value" format="24hr" />
    </v-menu>
  </v-text-field>
  <v-slider
    v-else-if="componentType === 'slider'"
    class="task-config-field task-config-field--slider"
    v-model="item.value"
    :label="translateAny(item.ui?.label)"
    :hint="translateAny(item.ui?.hint)"
    :color="themeColor"
    density="comfortable"
    persistent-hint
    v-bind="componentProps"
  />
  <v-textarea
    v-else-if="componentType === 'textarea'"
    class="task-config-field"
    v-model="item.value"
    :label="translateAny(item.ui?.label)"
    :hint="translateAny(item.ui?.hint)"
    :append-icon="item.ui?.resettable ? 'md:replay' : undefined"
    :prepend-inner-icon="item.ui?.readonly ? 'md:lock_outline' : undefined"
    :readonly="item.ui?.readonly"
    :color="themeColor"
    density="comfortable"
    persistent-hint
    auto-grow
    v-bind="componentProps"
    @click:append="resetValue"
  />
  <v-text-field
    v-else-if="componentType === 'number'"
    class="task-config-field"
    v-model.number="item.value"
    :label="translateAny(item.ui?.label)"
    :hint="translateAny(item.ui?.hint)"
    :append-icon="item.ui?.resettable ? 'md:replay' : undefined"
    :prepend-inner-icon="item.ui?.readonly ? 'md:lock_outline' : undefined"
    :readonly="item.ui?.readonly"
    :color="themeColor"
    type="number"
    density="comfortable"
    persistent-hint
    v-bind="componentProps"
    @click:append="resetValue"
  />
  <v-text-field
    v-else-if="componentType === 'text'"
    class="task-config-field"
    v-model="item.value"
    :label="translateAny(item.ui?.label)"
    :hint="translateAny(item.ui?.hint)"
    :append-icon="item.ui?.resettable ? 'md:replay' : undefined"
    :prepend-inner-icon="item.ui?.readonly ? 'md:lock_outline' : undefined"
    :readonly="item.ui?.readonly"
    :color="themeColor"
    density="comfortable"
    persistent-hint
    v-bind="componentProps"
    @click:append="resetValue"
  />
  <v-alert
    v-else
    type="warning"
    variant="tonal"
    density="comfortable"
    class="task-config-field"
  >
    {{ translateKey('config.componentNotRegistered', { component: componentType }) }}
  </v-alert>
</template>

<style scoped>
.task-config-field {
  margin-bottom: 12px;
}

.task-config-field--slider {
  padding-inline: 8px;
}
</style>
