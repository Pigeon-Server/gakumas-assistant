<script setup>
import app from "@/main.js";
import {useAppStore} from "@/stores/app.ts";
import {computed} from "vue";
import { translateAny, translateKey, translateOptionTitle } from '@/scripts/i18n/translate'

const props = defineProps({
  task: Object,
  task_name: String,
})

const store = useAppStore()
let task_config = store.get_task_config(props.task_name)

const themeColor = app.config.globalProperties.$theme.color

const scenarioOptions = computed(() => task_config.scenario.ui?.options || [])

const hajimeDifficultyOptions = computed(() => task_config.difficulty.ui?.options || [])

const niaDifficultyOptions = computed(() => task_config.nia_difficulty.ui?.options || [])

const formationModeOptions = computed(() => task_config.support_card_mode.ui?.options || [])

const configReady = computed(() => Boolean(task_config.value?.scenario))

// 兼容旧配置：target_idol_card_name → target_idol_card_id
const idolCardField = computed(() =>
  task_config.value?.target_idol_card_id ?? task_config.value?.target_idol_card_name ?? null
)

const showHajimeDifficulty = computed(() => task_config.value?.scenario?.value === "hajime")
const showNiaDifficulty = computed(() => task_config.value?.scenario?.value === "nia")
const showSupportPreset = computed(() => task_config.value?.support_card_mode?.value === "preset")
const showMemoryPreset = computed(() => task_config.value?.memory_mode?.value === "preset")
</script>

<template>
  <v-form v-if="configReady" v-auto-save="() => store.save_task_config(task_name)">
    <v-row dense>
      <v-col cols="12">
        <v-select
          :label="translateAny(task_config.scenario.ui.label)"
          :hint="translateAny(task_config.scenario.ui.hint)"
          :color="themeColor"
          :item-color="themeColor"
          :items="scenarioOptions"
          :item-title="translateOptionTitle"
          item-value="value"
          density="comfortable"
          persistent-hint
          v-model="task_config.scenario.value"
        >
          <template #item="{ props: optionProps, item: optionItem }">
            <v-list-item v-bind="optionProps" :title="translateAny(optionItem.raw?.title)" />
          </template>
          <template #selection="{ item: optionItem }">
            <span>{{ translateAny(optionItem.raw?.title) }}</span>
          </template>
        </v-select>
      </v-col>
      <v-col cols="12" v-if="showHajimeDifficulty">
        <v-select
          :label="translateAny(task_config.difficulty.ui.label)"
          :hint="translateAny(task_config.difficulty.ui.hint)"
          :color="themeColor"
          :item-color="themeColor"
          :items="hajimeDifficultyOptions"
          :item-title="translateOptionTitle"
          item-value="value"
          density="comfortable"
          persistent-hint
          v-model="task_config.difficulty.value"
        >
          <template #item="{ props: optionProps, item: optionItem }">
            <v-list-item v-bind="optionProps" :title="translateAny(optionItem.raw?.title)" />
          </template>
          <template #selection="{ item: optionItem }">
            <span>{{ translateAny(optionItem.raw?.title) }}</span>
          </template>
        </v-select>
      </v-col>
      <v-col cols="12" v-if="showNiaDifficulty">
        <v-select
          :label="translateAny(task_config.nia_difficulty.ui.label)"
          :hint="translateAny(task_config.nia_difficulty.ui.hint)"
          :color="themeColor"
          :item-color="themeColor"
          :items="niaDifficultyOptions"
          :item-title="translateOptionTitle"
          item-value="value"
          density="comfortable"
          persistent-hint
          v-model="task_config.nia_difficulty.value"
        >
          <template #item="{ props: optionProps, item: optionItem }">
            <v-list-item v-bind="optionProps" :title="translateAny(optionItem.raw?.title)" />
          </template>
          <template #selection="{ item: optionItem }">
            <span>{{ translateAny(optionItem.raw?.title) }}</span>
          </template>
        </v-select>
      </v-col>
      <v-col cols="12" v-if="idolCardField">
        <idol_card_browser :data="task_config"/>
      </v-col>
      <v-col cols="12">
        <v-select
          :label="translateAny(task_config.support_card_mode.ui.label)"
          :hint="translateAny(task_config.support_card_mode.ui.hint)"
          :color="themeColor"
          :item-color="themeColor"
          :items="formationModeOptions"
          :item-title="translateOptionTitle"
          item-value="value"
          density="comfortable"
          persistent-hint
          v-model="task_config.support_card_mode.value"
        >
          <template #item="{ props: optionProps, item: optionItem }">
            <v-list-item v-bind="optionProps" :title="translateAny(optionItem.raw?.title)" />
          </template>
          <template #selection="{ item: optionItem }">
            <span>{{ translateAny(optionItem.raw?.title) }}</span>
          </template>
        </v-select>
      </v-col>
      <v-col cols="12" v-if="showSupportPreset">
        <v-text-field
          :label="translateAny(task_config.support_card_preset_index.ui.label)"
          :hint="translateAny(task_config.support_card_preset_index.ui.hint)"
          :color="themeColor"
          type="number"
          density="comfortable"
          persistent-hint
          v-model.number="task_config.support_card_preset_index.value"
        />
      </v-col>
      <v-col cols="12">
        <v-select
          :label="translateAny(task_config.memory_mode.ui.label)"
          :hint="translateAny(task_config.memory_mode.ui.hint)"
          :color="themeColor"
          :item-color="themeColor"
          :items="formationModeOptions"
          :item-title="translateOptionTitle"
          item-value="value"
          density="comfortable"
          persistent-hint
          v-model="task_config.memory_mode.value"
        >
          <template #item="{ props: optionProps, item: optionItem }">
            <v-list-item v-bind="optionProps" :title="translateAny(optionItem.raw?.title)" />
          </template>
          <template #selection="{ item: optionItem }">
            <span>{{ translateAny(optionItem.raw?.title) }}</span>
          </template>
        </v-select>
      </v-col>
      <v-col cols="12" v-if="showMemoryPreset">
        <v-text-field
          :label="translateAny(task_config.memory_preset_index.ui.label)"
          :hint="translateAny(task_config.memory_preset_index.ui.hint)"
          :color="themeColor"
          type="number"
          density="comfortable"
          persistent-hint
          v-model.number="task_config.memory_preset_index.value"
        />
      </v-col>
      <v-col cols="12">
        <v-switch
          :label="translateAny(task_config.use_rental.ui.label)"
          :hint="translateAny(task_config.use_rental.ui.hint)"
          :color="themeColor"
          density="comfortable"
          persistent-hint
          v-model="task_config.use_rental.value"
        />
      </v-col>
      <v-col cols="12">
        <v-switch
          :label="translateAny(task_config.use_boost_items.ui.label)"
          :hint="translateAny(task_config.use_boost_items.ui.hint)"
          :color="themeColor"
          density="comfortable"
          persistent-hint
          v-model="task_config.use_boost_items.value"
        />
      </v-col>
      <v-col cols="12">
        <v-switch
          :label="translateAny(task_config.resume_interrupted.ui.label)"
          :hint="translateAny(task_config.resume_interrupted.ui.hint)"
          :color="themeColor"
          density="comfortable"
          persistent-hint
          v-model="task_config.resume_interrupted.value"
        />
      </v-col>
    </v-row>
  </v-form>
  <div v-else class="pa-4 text-body-2 text-medium-emphasis">
    {{ translateKey('common.loading') }}…{{ translateKey('config.loadingHint') }}
  </div>
</template>

<style scoped>
.v-row {
  padding-top: 15px;
}

.v-text-field,
.v-select,
.v-autocomplete,
.v-switch,
.v-slider,
.v-textarea {
  margin-bottom: 12px;
}
</style>
