<script>
import { translateAny as resolveLocalizedText, translateKey as resolveTranslationKey } from '@/scripts/i18n/translate'
export default {
  name: "TaskErrorReportDialog",
  props: {
    title: {
      type: [String, Object],
      required: false,
      default: "",
    },
    task_id: {
      type: String,
      required: false,
      default: "",
    },
    task_name: {
      type: [String, Object],
      required: false,
      default: "",
    },
    error_type: {
      type: String,
      required: false,
      default: "",
    },
    error_message: {
      type: [String, Object],
      required: false,
      default: "",
    },
    package_path: {
      type: String,
      required: false,
      default: "",
    },
    package_download_url: {
      type: String,
      required: false,
      default: "",
    },
    dump_dir: {
      type: String,
      required: false,
      default: "",
    },
    feedback: {
      type: Object,
      required: false,
      default: () => ({
        github_issues: "",
        qq_group: "",
      }),
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
  data() {
    return {
      flag: true,
      copying: false,
    }
  },
  computed: {
    dialogTitleText() {
      return resolveLocalizedText(this.title) || resolveTranslationKey('dialogs.taskError.title')
    },
    taskLabel() {
      return resolveLocalizedText(this.task_name) || this.task_id || resolveTranslationKey('dialogs.taskError.unknownTask')
    },
    taskDescriptionText() {
      return resolveTranslationKey('dialogs.taskError.description', { task: this.taskLabel })
    },
    exceptionText() {
      if (!this.error_type && !this.error_message) {
        return ""
      }
      return resolveTranslationKey('dialogs.taskError.exception', {
        errorType: this.error_type || 'UnknownError',
        errorMessage: resolveLocalizedText(this.error_message),
      })
    },
    downloadLogLabel() {
      return resolveTranslationKey('dialogs.taskError.downloadLog')
    },
    copyQqGroupLabel() {
      return resolveTranslationKey('dialogs.taskError.copyQqGroup')
    },
    openGithubLabel() {
      return resolveTranslationKey('dialogs.taskError.openGithub')
    },
    acknowledgeLabel() {
      return resolveTranslationKey('dialogs.taskError.acknowledge')
    },
    githubIssuesURL() {
      return this.feedback?.github_issues || "https://github.com/Pigeon-Server/gakumas-assistant/issues"
    },
    qqGroupNumber() {
      return this.feedback?.qq_group || "328346267"
    },
    canDownloadPackage() {
      return Boolean(this.package_download_url)
    },
    packageDownloadURL() {
      if (!this.package_download_url) {
        return ""
      }
      if (/^https?:\/\//i.test(this.package_download_url)) {
        return this.package_download_url
      }
      return new URL(this.package_download_url, window.location.origin).toString()
    },
  },
  methods: {
    translateAny(value) {
      return resolveLocalizedText(value)
    },
    async copyText(value) {
      if (!value || this.copying) {
        return
      }
      this.copying = true
      try {
        if (navigator?.clipboard?.writeText) {
          await navigator.clipboard.writeText(value)
        } else {
          const input = document.createElement("textarea")
          input.value = value
          input.setAttribute("readonly", "true")
          input.style.position = "fixed"
          input.style.left = "-9999px"
          document.body.appendChild(input)
          input.select()
          document.execCommand("copy")
          document.body.removeChild(input)
        }
      } finally {
        this.copying = false
      }
    },
    openGithubIssues() {
      if (!this.githubIssuesURL) {
        return
      }
      window.open(this.githubIssuesURL, "_blank", "noopener,noreferrer")
    },
    downloadPackage() {
      if (!this.packageDownloadURL) {
        return
      }
      const link = document.createElement("a")
      link.href = this.packageDownloadURL
      link.rel = "noopener noreferrer"
      link.style.display = "none"
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    },
    closeDialog() {
      this.confirm({ reason: "acknowledged" })
    },
  },
}
</script>

<template>
  <v-dialog
    v-model="flag"
    width="calc(100vw - 24px)"
    max-width="680"
    persistent
    @close="closeDialog"
  >
    <v-card>
      <v-card-title class="text-error">{{ dialogTitleText }}</v-card-title>
      <v-card-text class="task-error-dialog__content">
        <p class="task-error-dialog__line">
          {{ taskDescriptionText }}
        </p>
        <p class="task-error-dialog__line" v-if="exceptionText">
          {{ exceptionText }}
        </p>
      </v-card-text>
      <v-card-actions class="task-error-dialog__actions">
        <v-btn
          color="primary"
          :disabled="!canDownloadPackage"
          @click="downloadPackage"
        >
          {{ downloadLogLabel }}
        </v-btn>
        <v-btn
          color="primary"
          @click="copyText(qqGroupNumber)"
        >
          {{ copyQqGroupLabel }}
        </v-btn>
        <v-btn
          color="info"
          @click="openGithubIssues"
        >
          {{ openGithubLabel }}
        </v-btn>
        <v-btn color="error" @click="closeDialog">{{ acknowledgeLabel }}</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.task-error-dialog__content {
  display: grid;
  gap: 8px;
  line-height: 1.5;
}

.task-error-dialog__line {
  margin: 0;
}

.task-error-dialog__path {
  word-break: break-all;
}

.task-error-dialog__actions {
  flex-wrap: wrap;
  gap: 8px;
}
</style>
