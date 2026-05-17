<script setup lang="ts">
import MarkdownIt from "markdown-it";
import { computed } from "vue";
import aboutMarkdownSource from "@/assets/about.md?raw";

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

const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  typographer: true,
});

const renderedMarkdown = computed(() => markdown.render(aboutMarkdownSource));
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
      <article class="about_markdown" v-html="renderedMarkdown" />
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

.about_drawer__content {
  flex: 1 1 auto;
  min-height: 0;
  overflow: auto;
  padding: 12px;
  background: rgb(var(--v-theme-background));
}

.about_markdown {
  box-sizing: border-box;
  width: 100%;
  max-width: 100%;
  padding: 18px !important;
  border-radius: 16px;
  overflow-wrap: anywhere;
  color: rgb(var(--v-theme-on-surface));
  background: rgb(var(--v-theme-surface));
  border: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
  line-height: 1.7;
}

.about_markdown :deep(*) {
  box-sizing: border-box;
}

.about_markdown :deep(h1),
.about_markdown :deep(h2),
.about_markdown :deep(h3),
.about_markdown :deep(h4),
.about_markdown :deep(h5),
.about_markdown :deep(h6) {
  margin-top: 1.4em;
  margin-bottom: 0.7em;
  color: inherit;
  font-weight: 700;
  line-height: 1.3;
}

.about_markdown :deep(h1:first-child),
.about_markdown :deep(h2:first-child),
.about_markdown :deep(h3:first-child) {
  margin-top: 0;
}

.about_markdown :deep(h1),
.about_markdown :deep(h2) {
  padding-bottom: 0.35em;
  border-bottom: 1px solid rgba(var(--v-theme-on-surface), 0.1);
}

.about_markdown :deep(p),
.about_markdown :deep(blockquote),
.about_markdown :deep(pre),
.about_markdown :deep(table) {
  margin: 0 0 1em;
}

.about_markdown :deep(ul),
.about_markdown :deep(ol) {
  margin-top: 0;
  margin-bottom: 1em;
  margin-left: 0 !important;
  padding-inline-start: 1.8em !important;
}

.about_markdown :deep(li) {
  margin-left: 0 !important;
  list-style-position: outside;
}

.about_markdown :deep(li::marker) {
  color: rgba(var(--v-theme-on-surface), 0.55);
}

.about_markdown :deep(a) {
  color: rgb(var(--v-theme-primary));
  text-decoration: none;
}

.about_markdown :deep(a:hover) {
  text-decoration: underline;
}

.about_markdown :deep(strong) {
  color: inherit;
  font-weight: 700;
}

.about_markdown :deep(hr) {
  height: 1px;
  margin: 1.5em 0;
  border: 0;
  background: rgba(var(--v-theme-on-surface), 0.1);
}

.about_markdown :deep(blockquote) {
  margin-left: 0;
  padding: 12px 16px;
  color: rgba(var(--v-theme-on-surface), 0.82);
  background: rgba(var(--v-theme-primary), 0.05);
  border-left: 4px solid rgba(var(--v-theme-primary), 0.45);
  border-radius: 0 12px 12px 0;
}

.about_markdown :deep(code) {
  padding: 0.15em 0.4em;
  color: inherit;
  font-size: 0.92em;
  background: rgba(var(--v-theme-on-surface), 0.06);
  border-radius: 6px;
}

.about_markdown :deep(pre) {
  overflow: auto;
  padding: 14px 16px;
  background: rgba(var(--v-theme-on-surface), 0.05);
  border: 1px solid rgba(var(--v-theme-on-surface), 0.08);
  border-radius: 12px;
}

.about_markdown :deep(pre code) {
  padding: 0;
  background: transparent;
  border-radius: 0;
}

.about_markdown :deep(table) {
  display: block;
  width: 100%;
  overflow-x: auto;
  border-collapse: collapse;
}

.about_markdown :deep(th),
.about_markdown :deep(td) {
  padding: 10px 12px;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.1);
}

.about_markdown :deep(th) {
  font-weight: 700;
  text-align: left;
  background: rgba(var(--v-theme-on-surface), 0.05);
}

.about_markdown :deep(img) {
  max-width: 100%;
  border-radius: 12px;
}
</style>
