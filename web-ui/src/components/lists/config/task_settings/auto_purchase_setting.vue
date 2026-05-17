<script setup>
import {useAppStore} from "@/stores/app.ts";
import { translateAny } from '@/scripts/i18n/translate'
import SelectItem from "@/components/lists/config/task_settings/purchase_setting/select_item.vue";

const props = defineProps({
  task: Object,
  task_name: String,
})

const store = useAppStore()

let task_config = store.get_task_config(props.task_name)
</script>

<template>
  <v-form v-auto-save="() => store.save_task_config(task_name)">
    <v-row dense>
      <v-col cols="12">
        <v-switch
          :label="translateAny(task_config.weekly_gift.ui.label)"
          :hint="translateAny(task_config.weekly_gift.ui.hint)"
          persistent-hint
          clearable
          density="comfortable"
          v-model="task_config.weekly_gift.value"
        />
      </v-col>
      <v-col cols="12">
        <v-switch
          :label="translateAny(task_config.refresh_shop.ui.label)"
          :hint="translateAny(task_config.refresh_shop.ui.hint)"
          persistent-hint
          clearable
          density="comfortable"
          v-model="task_config.refresh_shop.value"
        />
      </v-col>
      <v-col cols="12" v-if="task_config.refresh_shop.value">
        <v-switch
          :label="translateAny(task_config.use_gem_refresh.ui.label)"
          :hint="translateAny(task_config.use_gem_refresh.ui.hint)"
          clearable
          density="comfortable"
          v-model="task_config.use_gem_refresh.value"
        />
      </v-col>
      <v-col cols="12">
        <SelectItem :data="task_config"/>
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
  margin-bottom: 12px;
}
</style>
