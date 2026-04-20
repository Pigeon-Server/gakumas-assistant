<script setup lang="ts">
import "github-markdown-css/github-markdown-dark.css";
import MarkdownIt from "markdown-it";
import { computed } from "vue";
import aboutMarkdownRaw from "@/assets/about.md?raw";

const props = defineProps({
  modelValue: {
    type: Boolean,
    default: true,
  },
  temporary: {
    type: Boolean,
    default: false,
  },
  disableTransition: {
    type: Boolean,
    default: false,
  },
  width: {
    type: [Number, String],
    default: 400,
  },
});
const emit = defineEmits(["update:modelValue"]);

const drawerValue = computed({
  get: () => (props.temporary ? props.modelValue : true),
  set: value => emit("update:modelValue", value),
});

const FALLBACK_MARKDOWN = `# Gakumas Assistant

学园偶像大师自动化助手。
`;

const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  typographer: true,
});

const markdownSource = computed(() => aboutMarkdownRaw?.trim() || FALLBACK_MARKDOWN);
const renderedMarkdown = computed(() => markdown.render(markdownSource.value));
</script>

<template>
  <v-navigation-drawer
    v-model="drawerValue"
    :permanent="!temporary"
    :temporary="temporary"
    :scrim="temporary"
    :width="width"
    :class="['about_drawer', { 'about_drawer--instant': disableTransition }]"
  >
    <div class="about_drawer__content">
      <article class="about_markdown markdown-body" v-html="renderedMarkdown" />
    </div>
  </v-navigation-drawer>
</template>

<style scoped>
.about_drawer {
  max-width: 100%;
}

.about_drawer :deep(.v-navigation-drawer__content) {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.about_drawer--instant {
  transition: none !important;
}

.about_drawer--instant :deep(.v-navigation-drawer__content) {
  transition: none !important;
}

.about_drawer__title_card {
  flex: 0 0 auto;
  min-height: 72px;
  display: flex;
  align-items: center;
  padding: 12px 16px;
}

.about_drawer__title_wrap {
  display: grid;
  gap: 4px;
}


.about_drawer__subtitle {
  color: rgba(255, 255, 255, 0.72);
}

.about_drawer__content {
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
  padding: 12px;
}

.about_markdown {
  box-sizing: border-box;
  width: 100%;
  max-width: 100%;
  padding: 18px !important;
  border-radius: 12px;
  overflow-wrap: anywhere;
}

.about_markdown :deep(ul),
.about_markdown :deep(ol) {
  margin-left: 0 !important;
  padding-inline-start: 1.8em !important;
}

.about_markdown :deep(li) {
  margin-left: 0 !important;
  list-style-position: outside;
}

.about_markdown :deep(li::marker) {
  color: #8b949e;
}

.markdown-body {
  background-color: unset !important;
}
</style>
