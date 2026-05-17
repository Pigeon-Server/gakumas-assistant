<script setup>
import apis from "@/scripts/apis.js";
import message from "@/scripts/utils/message.js";
import { translateAny, translateKey } from '@/scripts/i18n/translate'

const props = defineProps({
  data: Object
})
</script>

<template>
  <v-list-item>
    <v-text-field
      :label="translateAny(props.data.base.game_window_name.ui.label)"
      :hint="translateAny(props.data.base.game_window_name.ui.hint)"
      append-icon="md:replay"
      v-model="props.data.base.game_window_name.value"
      @click:append="props.data.base.game_window_name.value = props.data.base.game_window_name.default_value"
      persistent-hint
    />
  </v-list-item>
  <v-list-item>
    <v-text-field
      :label="translateAny(props.data.dmm_player.game_exe_path.ui.label)"
      :hint="translateAny(props.data.dmm_player.game_exe_path.ui.hint)"
      v-model="props.data.dmm_player.game_exe_path.value"
      persistent-hint
    />
  </v-list-item>
  <v-list-item>
    <v-text-field
      :label="translateAny(props.data.dmm_player.viewer_id.ui.label)"
      :hint="translateAny(props.data.dmm_player.viewer_id.ui.hint)"
      v-model="props.data.dmm_player.viewer_id.value"
      persistent-hint
      disabled
    />
  </v-list-item>
  <v-list-item>
    <v-text-field
      :label="translateAny(props.data.dmm_player.open_id.ui.label)"
      :hint="translateAny(props.data.dmm_player.open_id.ui.hint)"
      v-model="props.data.dmm_player.open_id.value"
      persistent-hint
      disabled
    />
  </v-list-item>
  <v-list-item>
    <v-text-field
      :label="translateAny(props.data.dmm_player.pf_token.ui.label)"
      :hint="translateAny(props.data.dmm_player.pf_token.ui.hint)"
      v-model="props.data.dmm_player.pf_token.value"
      persistent-hint
      disabled
    />
  </v-list-item>
  <v-list-item>
    <v-btn
      block
      append-icon="md:refresh"
      @click="async ()=> {
        await apis.refresh_ddm_player_token()
        props.data = (await apis.get_config()).data
        await message.showSuccess(translateKey('settings.refreshLaunchArgsSuccess'))
      }">
      {{ translateKey('settings.refreshLaunchArgs') }}
    </v-btn>
  </v-list-item>
</template>

<style scoped>

</style>
