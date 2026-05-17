<script setup>
import {useAppStore} from "@/stores/app.ts";
import support_card_whitelist from "@/components/lists/config/task_settings/enhancement_setting/support_card_whitelist.vue";
import { translateAny } from '@/scripts/i18n/translate'

const props = defineProps({
  task: Object,
  task_name: String,
})

const store = useAppStore()

let task_config = store.get_task_config(props.task_name)

const rarity_max_levels = {
  r: 40,
  sr: 50,
  ssr: 60,
}
</script>

<template>
  <v-form v-auto-save="() => store.save_task_config(task_name)">
    <v-row dense>
      <!-- SSR -->
      <v-col cols="12">
        <v-switch
          :label="translateAny(task_config.enhance_ssr.ui.label)"
          :hint="translateAny(task_config.enhance_ssr.ui.hint)"
          persistent-hint
          density="comfortable"
          v-model="task_config.enhance_ssr.value"
        />
      </v-col>
      <v-col cols="12" v-if="task_config.enhance_ssr.value">
        <v-slider
          :label="translateAny(task_config.enhance_ssr_max_level.ui.label)"
          :hint="translateAny(task_config.enhance_ssr_max_level.ui.hint)"
          v-model="task_config.enhance_ssr_max_level.value"
          :min="1"
          :max="rarity_max_levels.ssr"
          :step="1"
          thumb-label="always"
          density="comfortable"
        />
      </v-col>

      <!-- SR -->
      <v-col cols="12">
        <v-switch
          :label="translateAny(task_config.enhance_sr.ui.label)"
          :hint="translateAny(task_config.enhance_sr.ui.hint)"
          persistent-hint
          density="comfortable"
          v-model="task_config.enhance_sr.value"
        />
      </v-col>
      <v-col cols="12" v-if="task_config.enhance_sr.value">
        <v-slider
          :label="translateAny(task_config.enhance_sr_max_level.ui.label)"
          :hint="translateAny(task_config.enhance_sr_max_level.ui.hint)"
          v-model="task_config.enhance_sr_max_level.value"
          :min="1"
          :max="rarity_max_levels.sr"
          :step="1"
          thumb-label="always"
          density="comfortable"
        />
      </v-col>

      <!-- R -->
      <v-col cols="12">
        <v-switch
          :label="translateAny(task_config.enhance_r.ui.label)"
          :hint="translateAny(task_config.enhance_r.ui.hint)"
          persistent-hint
          density="comfortable"
          v-model="task_config.enhance_r.value"
        />
      </v-col>
      <v-col cols="12" v-if="task_config.enhance_r.value">
        <v-slider
          :label="translateAny(task_config.enhance_r_max_level.ui.label)"
          :hint="translateAny(task_config.enhance_r_max_level.ui.hint)"
          v-model="task_config.enhance_r_max_level.value"
          :min="1"
          :max="rarity_max_levels.r"
          :step="1"
          thumb-label="always"
          density="comfortable"
        />
      </v-col>

      <!-- 白名单模式 -->
      <v-col cols="12">
        <v-divider class="my-2" />
      </v-col>

      <!-- 上限解放 -->
      <v-col cols="12">
        <v-switch
          :label="translateAny(task_config.auto_limit_break.ui.label)"
          :hint="translateAny(task_config.auto_limit_break.ui.hint)"
          persistent-hint
          density="comfortable"
          v-model="task_config.auto_limit_break.value"
        />
      </v-col>

      <!-- サポート変換 -->
      <v-col cols="12">
        <v-switch
          :label="translateAny(task_config.auto_convert.ui.label)"
          :hint="translateAny(task_config.auto_convert.ui.hint)"
          persistent-hint
          density="comfortable"
          v-model="task_config.auto_convert.value"
        />
      </v-col>

      <v-col cols="12">
        <v-divider class="my-2" />
      </v-col>

      <v-col cols="12">
        <v-switch
          :label="translateAny(task_config.whitelist_mode.ui.label)"
          :hint="translateAny(task_config.whitelist_mode.ui.hint)"
          persistent-hint
          density="comfortable"
          v-model="task_config.whitelist_mode.value"
        />
      </v-col>
      <v-col cols="12" v-if="task_config.whitelist_mode.value">
        <support_card_whitelist :data="task_config" />
      </v-col>
    </v-row>
  </v-form>
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
  margin-bottom: 4px;
}
</style>
