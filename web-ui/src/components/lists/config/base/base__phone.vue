<script setup>
import app from "@/main.js";
import { translateAny } from '@/scripts/i18n/translate'

const props = defineProps({
  data: Object
})
</script>

<template>
  <base__adb_connect_mode :data="props.data"/>
  <div v-if="props.data.base.adb_connect_mode.value === 'Network'">
    <v-list-item>
      <v-text-field
        :label="translateAny(props.data.base.adb_host.ui.label)"
        :hint="translateAny(props.data.base.adb_host.ui.hint)"
        append-icon="md:replay"
        v-model="props.data.base.adb_host.value"
        @click:append="props.data.base.adb_host.value = props.data.base.adb_host.default_value"
        persistent-hint
      />
    </v-list-item>
    <v-list-item>
      <v-text-field
        :label="translateAny(props.data.base.adb_port.ui.label)"
        :hint="translateAny(props.data.base.adb_port.ui.hint)"
        type="number"
        append-icon="md:replay"
        v-model="props.data.base.adb_port.value"
        @click:append="props.data.base.adb_port.value = props.data.base.adb_port.default_value"
        persistent-hint
      />
    </v-list-item>
  </div>
  <div v-else>
    <base__adb_devices :data="props.data" :only_usb_device="true" />
  </div>
  <v-list-item>
    <v-select
      :label="translateAny(props.data.base.android_screen_capture_service.ui.label)"
      :hint="translateAny(props.data.base.android_screen_capture_service.ui.hint)"
      :items="['scrcpy', 'DroidCast', 'ADB']"
      :item-color="app.config.globalProperties.$theme.color"
      v-model="props.data.base.android_screen_capture_service.value"
      persistent-hint
    />
  </v-list-item>
  <v-list-item>
    <v-select
      :label="translateAny(props.data.base.android_touch_service.ui.label)"
      :hint="translateAny(props.data.base.android_touch_service.ui.hint)"
      :items="['maatouch', 'minitouch', 'scrcpy', 'ADB']"
      :item-color="app.config.globalProperties.$theme.color"
      v-model="props.data.base.android_touch_service.value"
      persistent-hint
    />
  </v-list-item>
  <v-list-item>
    <v-text-field
      :label="translateAny(props.data.base.game_package_name.ui.label)"
      :hint="translateAny(props.data.base.game_package_name.ui.hint)"
      append-icon="md:replay"
      v-model="props.data.base.game_package_name.value"
      @click:append="props.data.base.game_package_name.value = props.data.base.game_package_name.default_value"
      persistent-hint
    />
  </v-list-item>
</template>

<style scoped>

</style>
