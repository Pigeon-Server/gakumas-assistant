<script setup>
import app from "@/main.js";
import {useAppStore} from "@/stores/app.ts";
import { translateAny } from '@/scripts/i18n/translate'
import SelectChallengeOrder from "@/components/lists/config/task_settings/auto_contest/select_challenge_order.vue";

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
          :label="translateAny(task_config.auto_reconfigure_team_before_challenge.ui.label)"
          :hint="translateAny(task_config.auto_reconfigure_team_before_challenge.ui.hint)"
          :color="app.config.globalProperties.$theme.color"
          persistent-hint
          v-model="task_config.auto_reconfigure_team_before_challenge.value"
        />
      </v-col>
      <v-col cols="12">
        <SelectChallengeOrder :data="task_config"/>
      </v-col>
    </v-row>
  </v-form>
</template>

<style scoped>

</style>
