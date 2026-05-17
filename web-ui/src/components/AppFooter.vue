<template>
  <v-footer
    app
    class="app-footer"
  >
    <div v-if="showBrowserShutdown" class="app-footer__actions">
      <v-btn
        :loading="shutdownPending"
        color="warning"
        variant="tonal"
        prepend-icon="md:power_settings_new"
        @click="shutdownApp"
      >
        {{ translateKey('app.footer.exit') }}
      </v-btn>
    </div>
    <div class="app-footer__meta text-caption text-disabled">
      &copy; 2020-{{ (new Date()).getFullYear() }} <span class="d-none d-sm-inline-block">{{ translateKey('app.footer.teamName') }}</span>
      —
      <a
        class="text-decoration-none on-surface"
        href="https://github.com/Pigeon-Server/gakumas-assistant/"
        rel="noopener noreferrer"
        target="_blank"
      >
        {{ translateKey('app.footer.licenseLabel') }}
      </a>
    </div>
  </v-footer>
</template>

<script setup>
  import { computed, onBeforeUnmount, onMounted, ref } from "vue";

  import apis from "@/scripts/apis.js";
  import message from "@/scripts/utils/message.js";
  import { addWindowHostReadyListener, isWindowHostAvailable } from "@/scripts/utils/windowHost.js";
  import { translateKey } from "@/scripts/i18n/translate";

  const nativeWindowHost = ref(isWindowHostAvailable());
  const shutdownPending = ref(false);
  let removeWindowHostReadyListener = null;

  const showBrowserShutdown = computed(() => !nativeWindowHost.value);

  function syncWindowHostState() {
    nativeWindowHost.value = isWindowHostAvailable();
  }

  async function shutdownApp() {
    if (shutdownPending.value) {
      return;
    }
    shutdownPending.value = true;
    try {
      await apis.shutdown_app();
      message.showInfo(translateKey('app.footer.exiting'));
      window.setTimeout(() => {
        window.close();
      }, 300);
    } catch (error) {
      shutdownPending.value = false;
    }
  }

  onMounted(() => {
    syncWindowHostState();
    removeWindowHostReadyListener = addWindowHostReadyListener(syncWindowHostState);
  });

  onBeforeUnmount(() => {
    removeWindowHostReadyListener?.();
  });
</script>

<style scoped lang="sass">
  .app-footer
    min-height: 48px !important
    height: auto !important
    padding: 8px 16px
    justify-content: space-between

  .app-footer__actions
    display: flex
    align-items: center
    gap: 8px

  .app-footer__meta
    flex: 1
    text-align: right
    line-height: 1.4

  @media (max-width: 959px)
    .app-footer
      padding: 8px 12px
      flex-direction: column
      gap: 8px

    .app-footer__meta
      text-align: center

  .social-link :deep(.v-icon)
    color: rgba(var(--v-theme-on-background), var(--v-disabled-opacity))
    text-decoration: none
    transition: .2s ease-in-out

    &:hover
      color: rgba(25, 118, 210, 1)
</style>
