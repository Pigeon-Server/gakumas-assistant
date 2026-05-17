<script>
import { translateKey } from '@/scripts/i18n/translate'

export default {
  name: 'InputDialog',
  props: {
    title: {
      type: String,
      required: true,
    },
    description: {
      type: String,
      required: true,
    },
    persistent: {
      type: Boolean,
      required: false,
      default: null,
    },
    confirm: {
      type: Function,
      required: true,
    },
    confirm_text: {
      type: String,
      required: false,
      default: null,
    },
    close: {
      type: Function,
      required: true,
    },
    close_text: {
      type: String,
      required: false,
      default: null,
    }
  },
  data () {
    return {
      flag: true,
    }
  },
  computed: {
    resolvedConfirmText () {
      return this.confirm_text || translateKey('common.confirm')
    },
    resolvedCloseText () {
      return this.close_text || translateKey('common.cancel')
    },
  }
}
</script>

<template>
  <v-dialog
    v-model="flag"
    activator="parent"
    :persistent="persistent"
    width="calc(100vw - 24px)"
    max-width="400"
    @close="close()"
  >
    <v-card>
      <v-card-title>{{ title }}</v-card-title>
      <v-card-text>
        {{ description }}
      </v-card-text>
      <v-card-actions>
        <v-btn color="success" @click="confirm()">{{ resolvedConfirmText }}</v-btn>
        <v-btn color="error" @click="close()">{{ resolvedCloseText }}</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>

</style>
