<script setup>
import { computed, ref } from "vue";
import apis from "@/scripts/apis.js";
import { translateAny } from '@/scripts/i18n/translate'

const props = defineProps({
  data: Object,
  only_usb_device: {
    type: Boolean,
    default: false,
  }
})

const device = ref([])
const load_status = ref(false)
const load_message = ref("")
const device_hint = computed(() => {
  const baseHint = translateAny(props.data.base.adb_serial.ui.hint)
  return load_message.value ? `${baseHint} ${translateAny(load_message.value)}` : baseHint
})

function load_device_list() {
  load_status.value = false
  load_message.value = ""
  apis.get_all_adb_device(props.only_usb_device).then((res) => {
    device.value = res.data.devices || []
    load_message.value = res.data.message || ""
    load_status.value = true
  })
}

load_device_list()
</script>

<template>
  <v-list-item>
    <v-select
      :items="device"
      :loading="!load_status"
      v-model="props.data.base.adb_serial.value"
      append-icon="md:replay"
      @click:append="load_device_list"
      clearable
      :label="translateAny(props.data.base.adb_serial.ui.label)"
      :hint="device_hint"
      persistent-hint
    />
  </v-list-item>
</template>

<style scoped>

</style>
