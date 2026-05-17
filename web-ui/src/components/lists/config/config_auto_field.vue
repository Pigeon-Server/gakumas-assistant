<script setup>
import { computed } from "vue";

import app from "@/main.js";
import BaseAdbDevices from "@/components/lists/config/base/base__adb_devices.vue";
import BaseDisabledTaskList from "@/components/lists/config/base/base__disabled_task_list.vue";
import { formatLocalizedList, translateAny, translateKey, translateOptionTitle } from '@/scripts/i18n/translate'

const props = defineProps({
  config: {
    type: Object,
    required: true,
  },
  item: {
    type: Object,
    required: true,
  }
})

const themeColor = app.config.globalProperties.$theme.color
const selectItems = computed(() => props.item?.ui?.options || [])

const componentType = computed(() => {
  if (props.item?.ui?.component) {
    return props.item.ui.component
  }
  if (props.item?.data_type === "bool") {
    return "switch"
  }
  if (props.item?.ui?.options?.length) {
    return "select"
  }
  return "text"
})

const isTextField = computed(() => ["text", "number"].includes(componentType.value))
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

function cloneValue(value) {
  if (value === null || value === undefined) {
    return value
  }
  return JSON.parse(JSON.stringify(value))
}

function resetValue() {
  props.item.value = cloneValue(props.item.default_value)
}

function selectItemProps(option) {
  return {
    disabled: Boolean(option?.disabled),
  }
}
</script>

<template>
  <BaseDisabledTaskList v-if="componentType === 'disabled_tasks'" />
  <BaseAdbDevices v-else-if="componentType === 'adb_devices'" :data="config" :only_usb_device="true" />
  <v-list-item v-else-if="componentType === 'switch'">
    <v-switch
      class="config-auto-field config-auto-field--switch"
      v-model="item.value"
      :label="translateAny(item.ui?.label)"
      :hint="translateAny(item.ui?.hint)"
      :color="themeColor"
      density="comfortable"
      persistent-hint
    />
  </v-list-item>
  <v-list-item v-else-if="componentType === 'select'">
    <v-select
      class="config-auto-field"
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
  </v-list-item>
  <v-list-item v-else-if="componentType === 'time'">
    <v-text-field
      class="config-auto-field"
      v-model="item.value"
      :label="translateAny(item.ui?.label)"
      :hint="translateAny(item.ui?.hint)"
      :color="themeColor"
      prepend-inner-icon="md:schedule"
      density="comfortable"
      persistent-hint
      readonly
    >
      <v-menu
        :close-on-content-click="false"
        activator="parent"
        min-width="0"
      >
        <v-time-picker v-model="item.value" format="24hr" />
      </v-menu>
    </v-text-field>
  </v-list-item>
  <v-list-item v-else-if="isTextField">
    <v-text-field
      class="config-auto-field"
      v-model="item.value"
      :label="translateAny(item.ui?.label)"
      :hint="translateAny(item.ui?.hint)"
      :type="componentType === 'number' ? 'number' : 'text'"
      :append-icon="item.ui?.resettable ? 'md:replay' : undefined"
      :prepend-inner-icon="item.ui?.readonly ? 'md:lock_outline' : undefined"
      :readonly="item.ui?.readonly"
      :color="themeColor"
      density="comfortable"
      persistent-hint
      @click:append="resetValue"
    />
  </v-list-item>
</template>
