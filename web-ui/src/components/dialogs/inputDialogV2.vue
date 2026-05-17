<script>
  import { translateKey } from '@/scripts/i18n/translate'

  export default {
    name: 'InputDialog',
    props: {
      title: {
        type: String,
        required: true,
      },
      label: {
        type: String,
        required: false,
        default: '',
      },
      hint: {
        type: String,
        required: false,
        default: '',
      },
      type: {
        type: String,
        required: false,
        default: 'text',
      },
      persistent: {
        type: Boolean,
        required: false,
        default: null,
      },
      default_value: {
        type: [String, Number],
        required: false,
        default: '',
      },
      max: {
        type: Number,
        required: false,
        default: null,
      },
      min: {
        type: Number,
        required: false,
        default: null,
      },
      confirm: {
        type: Function,
        required: true,
      },
      close: {
        type: Function,
        required: true,
      },
    },
    computed: {
      resolvedConfirmText () {
        return translateKey('common.confirm')
      },
      resolvedCloseText () {
        return translateKey('common.cancel')
      },
    },
    data () {
      return {
        flag: true,
        value: null,
      }
    },
    watch: {
      value (newVal, oldVal) {
        if (newVal === oldVal || newVal === this.default_value) {
          return
        }
        if (this.type === 'text') {
          if (newVal === null) {
            return
          }
          if (newVal.length > this.max) {
            this.value = oldVal
          } else if (newVal.length < this.min) {
            this.value = oldVal
          }
        }
        if (this.type === 'number') {
          if (newVal > this.max) {
            this.value = this.max
          } else if (newVal < this.min) {
            this.value = this.min
          }
        }
      },
    },
    created () {
      if (this.default_value) {
        this.value = this.default_value
      }
    },
  }
</script>

<template>
  <v-dialog
    id="inputDialog"
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
        <v-text-field
          v-model="value"
          :hint="hint"
          :label="label"
          :type="type"
        />
      </v-card-text>
      <v-card-actions>
        <v-btn color="success" @click="confirm(value)">{{ resolvedConfirmText }}</v-btn>
        <v-btn color="error" @click="close()">{{ resolvedCloseText }}</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>

</style>
