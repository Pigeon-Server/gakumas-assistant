import { en as vuetifyEn } from 'vuetify/locale'
import zhHans from './zhHans'
import { deepMergeMessages } from '@/scripts/i18n/messageTools'

/**
 * English locale messages.
 */
const en = deepMergeMessages(zhHans, {
  $vuetify: vuetifyEn,
  app: {
    sections: {
      tasks: 'Tasks',
      settings: 'Settings',
      about: 'About',
    },
    footer: {
      exit: 'Exit App',
      exiting: 'The app is shutting down...',
      teamName: 'Pigeon Server Team',
      licenseLabel: 'GPLv3 License',
      closeLogger: 'Close logs',
      openLogger: 'Open logs',
      executionLog: 'Execution Log',
    },
    window: {
      minimize: 'Minimize',
      maximize: 'Maximize',
      restore: 'Restore',
      close: 'Close',
    },
    preferences: {
      languageMenu: 'Switch language',
      themeMenu: 'Switch theme',
    },
  },
  common: {
    confirm: 'Confirm',
    cancel: 'Cancel',
    save: 'Save Settings',
    reset: 'Restore Defaults',
    loading: 'Loading',
    notRun: 'Never Run',
    yes: 'Yes',
    no: 'No',
    unknown: 'Unknown',
    close: 'Close',
    refresh: 'Refresh',
    retry: 'Retry',
    auto: 'Auto',
    currentTask: 'Current Task',
    manualOnly: 'Manual Only',
    system: 'System',
    light: 'Light',
    dark: 'Dark',
  },
  dialogs: {
    resetSettingsTitle: 'Reset all settings?',
    resetSettingsText: 'Please proceed carefully. This will restore all settings, including task settings, to their defaults.',
    runFromTitle: 'Start from this task?',
    runFromDescription: 'The app will start from "{task}" and continue with the enabled automatic tasks in list order.',
    taskError: {
      title: 'Task execution failed',
      unknownTask: 'Unknown Task',
      description: 'Task {task} failed. You can report the error log through GitHub Issues or the QQ group to help us locate the problem faster.',
      exception: 'Exception: {errorType} {errorMessage}',
      downloadLog: 'Download log archive',
      copyQqGroup: 'Copy QQ group number',
      openGithub: 'Open GitHub feedback',
      acknowledge: 'Got it',
    },
    selector: {
      chooseCard: 'Select Card',
      selectCard: 'Select Card',
      cancelSelect: 'Cancel Selection',
      searchPlaceholder: 'Search card name (supports Chinese/Japanese/ID)',
      selectedCount: 'Selected {count}',
      totalCards: '{count} cards',
      noMatchedCards: 'No matching cards',
      clearSelection: 'Clear selection',
      confirmSingle: 'Confirm',
      confirmMulti: 'Confirm ({count})',
    },
  },
  websocket: {
    disconnected: 'Connection lost',
    reconnected: 'Server reconnected successfully',
    waitingServer: 'Waiting for server response.....',
    invalidBinaryFormat: 'Invalid binary format',
  },
  toolbar: {
    deviceNotReady: 'Device not ready',
    deviceRetryHint: '{message} After you click "Start", the app will try to reconnect to the device again.',
    waitingAction: 'Waiting',
    running: 'Script is running......',
    suspended: 'Script is suspended......',
    start: 'Start',
    stop: 'Stop',
    suspend: 'Suspend',
    resume: 'Resume',
    startQueued: 'Task queue is running',
    stopQueued: 'Stopping task queue',
    suspendedDone: 'Task suspended',
    resumedDone: 'Task resumed',
  },
  settings: {
    title: 'Script Settings',
    basicSection: 'Basic Settings',
    language: {
      label: 'Interface Language',
      hint: 'Switch the WebUI display language',
      option: {
        system: 'Follow System',
        zhHans: '简体中文',
        zhHant: '繁體中文',
        en: 'English',
        ja: '日本語',
      },
    },
    theme: {
      label: 'Interface Theme',
      option: {
        system: 'Follow System',
        light: 'Light Mode',
        dark: 'Dark Mode',
      },
    },
    saveSuccess: 'Settings saved',
    resetSuccess: 'Settings reset. Some changes may require a restart to take effect.',
    refreshLaunchArgs: 'Refresh launch parameters',
    refreshLaunchArgsSuccess: 'Launch parameters refreshed',
    resourceUpdate: {
      title: 'Resource Updates',
      bootstrapAction: 'Download required resources',
      applyAction: 'Update now',
      checkActionIdle: 'Check for updates',
      checkActionBusy: 'Checking',
      state: {
        pending: 'Not checked',
        checked: 'Checked',
        checking: 'Checking',
        downloading: 'Downloading',
        updating: 'Updating',
        updateAvailable: 'Update available',
        bootstrapPending: 'Pending download',
        error: 'Check failed',
      },
      headline: {
        bootstrapIdle: 'This package no longer bundles the game database and localization assets. They must be downloaded on first launch.',
        bootstrapRunning: 'Downloading the game database and localization assets required for first launch',
        updating: 'Syncing resource repositories and reloading the game database',
        checking: 'Checking upstream updates for GakumasTranslationData and gakumasu-diff',
        hasUpdate: 'A new resource repository version is available',
        lastError: 'The last check ended with an error',
        checked: 'The current resource repository state has been synced',
        idle: 'You can check manually or wait for startup / scheduled checks',
      },
      bootstrapNotice: 'The game database and localization assets must be downloaded before the first launch. Failed downloads will retry automatically, and initialization will continue after completion.',
      progressFallbackTitle: 'Processing resources',
      progressFallbackMessage: 'Syncing resources, please wait.',
      recentError: 'Recent error: {error}',
      pendingDialogTitle: 'Resources are required before first launch',
      pendingDialogDescription: 'This package no longer bundles the game database and localization assets. After confirmation, the app will download them automatically and continue initialization.',
      pendingDialogMeta: 'The game database and localization assets must be downloaded before first launch.',
      pendingDialogMetaItem: '{name} ({missing}/{required} files missing)',
      dialogAgree: 'Agree and start download',
      dialogLater: 'Later',
      downloadProgressTitle: 'Downloading resources',
    },
  },
  tasks: {
    title: 'Task List',
    blockedHint: 'You can review tasks and settings before the resource download finishes, but execution is not available yet.',
    taskName: 'Task ID: ',
    enabled: 'Enabled: ',
    lastRunTime: 'Last run: ',
    run: 'Run',
    runFrom: 'Run from here',
    disable: 'Disable',
    enable: 'Enable',
    settings: 'Task Settings',
    runBlockedTitle: 'Resources are not ready yet',
    runBusyTitle: 'A task queue is already running',
    runFromTitle: 'Run enabled automatic tasks starting from this task',
    status: {
      PENDING: 'Pending',
      RUNNING: 'Running',
      SUSPENDED: 'Suspended',
      SUCCESS: 'Completed',
      FAILED: 'Failed',
      CANCELED: 'Canceled',
      UNKNOWN: 'Unknown',
    },
    relativeTime: {
      justNow: 'Just now',
      secondsAgo: '{count}s ago',
      minutesAgo: '{count}m ago',
      hoursAgo: '{count}h ago',
      yesterday: 'Yesterday',
      daysAgo: '{count}d ago',
    },
  },
  resource: {
    progress: {
      step: 'Step {current}/{total}',
      attempt: 'Attempt {current}/{total}',
      retryInSeconds: 'Retry in {seconds}s',
      lastCheckedAt: 'Last checked: {time}',
      nextCheckAt: 'Next scheduled check: {time}',
      bootstrapDownloadHint: 'The game database and localization assets must be downloaded before first launch',
      checkingHint: 'Checking resource repository updates',
      updatingHint: 'Updating resource repositories',
      notCheckedYet: 'No resource repository check has been run yet',
      checkFailed: 'Resource check failed: {error}',
      checkHasUpdate: 'Updates are available for the resource repositories',
      checkUpToDate: 'Resource repositories are already up to date',
      updateFailed: 'Resource update failed: {error}',
      bootstrapCompleted: 'First-launch resources downloaded. The game database has been reloaded.',
      updateCompleted: 'Resource repositories updated. The game database has been reloaded.',
      bootstrapPromptSingle: 'The game database and localization assets must be downloaded before first launch. Progress will be shown and failures will retry automatically. Start now?',
      bootstrapPromptMultiple: 'The following resources are required before first launch: {repositories}. Progress will be shown and failures will retry automatically. Start now?',
      updatePromptSingle: 'Updates are available for the resource repositories. Update now and reload the game database?',
      updatePromptMultiple: 'Updates are available for the following resource repositories: {repositories}. Update now and reload the game database?',
      repoMissingItem: '{name} ({missing}/{required} files missing)',
      repositoryCommitRange: '{name} ({local} -> {remote})',
      updatePromptTitle: 'Resource repository updates found',
      updatePromptConfirm: 'Update now',
      updatePromptCancel: 'Later',
      checkCompleted: 'Resource repository check finished',
      checkCompletedWithErrors: 'Resource repository check finished, but some repositories failed: {error}',
      bootstrapRunning: 'Downloading resources required for first launch...',
      updateRunning: 'Updating resource repositories...',
      operationLocked: 'A resource repository check or update is already in progress',
      taskRunning: 'Resource repositories cannot be updated while tasks are running',
      upToDate: 'Resource repositories are already up to date',
      reloadingDatabase: 'Reloading the game database',
      reloadingDatabaseMessage: 'Resource download completed. Reloading the game database and related services.',
      reloadFailed: 'Failed to reload resources: {error}',
      repositoryError: '{error}',
      noError: '',
      resourcesMissing: 'Resource download is incomplete. Required runtime resources are still missing.',
      retrying: 'Retrying download for {repository}',
      retryingMessage: '{repository} download failed. It will retry automatically in {seconds} seconds.',
      retryExceeded: 'Retried automatically {limit} times. Last error: {error}',
      updatingRepository: 'Updating {repository}',
      updatingRepositoryWithGit: 'Updating {repository} with Git.',
      repositoryUpdated: '{repository} updated',
      repositorySynced: '{repository} has been synced to the latest version.',
      installingRepository: 'Installing {repository}',
      writingRepository: 'Writing {repository} to the local resource directory.',
      repositoryUpdatedToLatest: '{repository} has been updated to the latest version.',
      preparingRepositoryDownload: 'Preparing to download {repository}',
      preparingRepositoryDownloadMessage: 'Preparing to download {repository} from {url}.',
      downloadingRepository: 'Downloading {repository}',
      downloadingRepositoryFromGithub: 'Downloading the {repository} package from GitHub.',
      extractingRepository: 'Extracting {repository}',
      extractingRepositoryMessage: 'Extracting the {repository} package.',
      downloadingRepositoryWithGit: 'Downloading the {repository} package with Git.',
    },
  },
  backend: {
    api: {
      ok: 'OK',
      genericError: 'error',
      invalidTaskName: 'Task does not exist',
      taskConfigMissing: 'This task has no configuration.',
      shutdownStarted: 'The app is shutting down',
      resourceNotReady: 'The game database and localization assets must be downloaded before first launch. Please confirm the download in WebUI.',
      gameDatabaseNotReady: 'Game database resources are not ready',
      taskQueueStartFailed: 'Failed to start the task queue',
      taskStartFailed: 'Failed to start the task',
      runFromFailed: 'Failed to start from the current task',
      manualOnlyRunFromUnsupported: 'Manual-only tasks cannot be used as a queue starting point',
      noRunningTask: 'No task is currently running',
      noSuspendedTask: 'No task is currently suspended',
      suspendUnsupported: 'The current task does not support manual suspension',
      resumeUnsupported: 'The current task does not support manual resume',
      resumeBlockedByInsertedTask: 'A temporary inserted task is running, so the queue cannot be resumed now',
      taskFailurePackageMissing: 'The log archive does not exist or has expired. Retry the task and download it again.',
      refreshDmmTokenFailed: 'Failed to extract game launch parameters: {error}',
      imageDownloadDisabled: 'Game asset download is disabled. Please enable it in settings.',
      imageDownloadFeatureDisabledShort: 'Game asset download is disabled',
      objectManagerUnavailable: 'GkmasObjectManager is not ready. Please make sure the vendor/GkmasObjectManager submodule is initialized.',
      objectManagerUnavailableShort: 'GkmasObjectManager is not ready',
      downloadInProgress: 'A download is already running. Please wait.',
      supportCardThumbDownloadStarted: 'Started downloading support card thumbnails',
      supportCardFullDownloadStarted: 'Started downloading full-size support card images',
      downloadStarted: 'Download started',
      downloadAlreadyExists: 'Already exists',
      downloadAlreadyRunning: 'Download is already running',
      supportCardAutoDownloadStarted: 'Started automatic download of support card images',
      cardNotFound: 'Card not found: {cardId}',
    },
    app: {
      deviceInitializing: 'Initializing device...',
      deviceUnavailable: 'The current device is unavailable.',
      deviceReadyAutoDetected: 'An available device was detected automatically',
      deviceDisconnected: 'The device connection was lost',
      status: {
        ready: 'Device ready',
        initializing: 'Initializing device',
      },
    },
    device: {
      windows: {
        available: 'Available',
        unavailable: {
          non_windows: 'PC mode is only supported on Windows. Please use Phone mode on macOS or Linux.',
          import_error: 'Required Windows-only components for PC mode are not ready, usually because pywin32 is missing or broken. Re-run `pip install -r requirements.txt` and try again.',
          unknown: 'PC mode is currently unavailable.',
        },
      },
      mac: {
        available: 'Available',
        unavailable: {
          non_macos: 'MacPlayTools mode is only supported on macOS (Apple Silicon).',
          import_error: 'Required components for MacPlayTools mode are not ready: {error}',
          unknown: 'MacPlayTools mode is currently unavailable.',
        },
      },
    },
    task: {
      startManual: 'Started manual task: {task}',
      startFrom: 'Started subsequent tasks from here: {task}',
      enabled: 'Task enabled: {task}',
      disabled: 'Task disabled: {task}',
      names: {
        start_game: 'Start Game',
        get_expenditure: 'Collect Event Stamina',
        dispatch_work: 'Dispatch Work',
        get_gift: 'Collect Gifts/Mailbox',
        auto_purchase: 'Daily Shop Automation',
        auto_enhancement_support_card: 'Auto Enhance Support Cards',
        auto_contest: 'Daily Contest Automation',
        claim_task_rewards: 'Claim Task Rewards',
        claim_pass_rewards: 'Claim Pass Rewards',
        auto_producer: 'Auto Produce (Beta)',
        void_task: 'Test Task',
        refresh_skill_storage: 'Refresh Skill Card Storage',
        learn_support_card_clip: 'Refresh Support Card Storage',
        learn_idol_card_clip: 'Refresh Idol Card Storage',
      },
    },
    gameAsset: {
      downloading: 'Downloading {label}...',
      fetchingManifest: 'Fetching resource manifest...',
      fetchingManifestWithLabel: 'Fetching resource manifest ({label})...',
      searchingWithLabel: 'Searching {label} resources...',
      noMatchingResources: 'No {label} resources were found',
      noMatchingObjects: 'No matching objects were found',
      downloadCompleted: '{label} download completed: {downloaded} new images, {skipped} skipped (total cached: {total})',
      phaseCompleted: '{label}: {downloaded} new images, {skipped} skipped (total cached: {total})',
      downloadFailed: '{label} download failed',
      downloadFailedError: '{error}',
      dialogAssetsCompleted: 'All support-card related resources have finished downloading',
      bulkDownloadFailed: 'Resource download failed',
    },
    message: {
      gameNotForegroundRetry: 'The game is not running in the foreground. Switch back to the game and try again.',
      gameNotStartedRetry: 'The game is not running. Start it manually and try again.',
      gameNotForegroundStart: 'The game is not running in the foreground. Start it manually and try again.',
      startLearningIdolClip: 'Starting idol card CLIP learning. The bottom carousel will be traversed card by card.',
    },
    config: {
      section: {
        base: 'Basic Settings',
        dmm_player: 'DMM Player',
      },
      base: {
        run_mode: {
          label: 'Run Mode',
          hint: 'Execution mode for the script (requires restart)',
          option: {
            pc: 'Desktop (DMM)',
            phone: 'Phone',
            mac_play_tools: 'macOS PlayCover',
          },
          disabledReason: {
            pcWindowsOnly: 'PC / DMM mode is only supported on Windows.',
            macOnly: 'MacPlayTools mode is only supported on macOS (Apple Silicon).',
          },
        },
        ocr_backend: {
          label: 'OCR Backend',
          hint: 'auto: prefer Vision on macOS and RapidOCR elsewhere; falls back to RapidOCR on failure (requires restart)',
          option: {
            auto: 'Auto',
            rapidocr: 'RapidOCR',
            vision: 'Vision (native macOS OCR)',
          },
          disabledReason: {
            visionMacOnly: 'Vision OCR is only available on macOS.',
          },
        },
        adb_connect_mode: {
          label: 'ADB Connection Mode',
          hint: 'Connection mode for Android Debug Bridge. USB is recommended for phones; emulators can use network connection. Restart required after changes.',
          option: {
            network: {
              title: 'Network',
            },
            u_s_b: {
              title: 'USB',
            },
          },
        },
        adb_host: {
          label: 'ADB Host',
          hint: 'IP address of Android Debug Bridge. Emulators usually use 127.0.0.1.',
        },
        adb_port: {
          label: 'ADB Port',
          hint: 'Port of Android Debug Bridge. The default is 5555; Android 11 and above may use a system-assigned random port.',
        },
        adb_serial: {
          label: 'USB-connected ADB Device',
          hint: 'Select a device connected over USB. If no device is found, try refreshing the list.',
        },
        android_screen_capture_service: {
          label: 'ADB Screenshot Method',
          hint: 'scrcpy / DroidCast usually have lower latency than plain ADB screenshots. For scrcpy, place the official scrcpy-server release file in bin.',
          option: {
            scrcpy: {
              title: 'scrcpy',
            },
            droid_cast: {
              title: 'DroidCast',
            },
            a_d_b: {
              title: 'ADB',
            },
          },
        },
        android_touch_service: {
          label: 'ADB Touch Method',
          hint: 'Available options: MaaTouch / minitouch / scrcpy. MaaTouch requires the official build output in bin/maatouch or a workflow-generated build. minitouch requires the official build output in bin/minitouch and currently supports Android 9 and below only.',
          option: {
            maatouch: {
              title: 'MaaTouch',
            },
            minitouch: {
              title: 'minitouch',
            },
            scrcpy: {
              title: 'scrcpy',
            },
            a_d_b: {
              title: 'ADB',
            },
          },
        },
        auto_start_game: {
          label: 'Auto Start Game',
          hint: 'Automatically start the game when it is not running.',
        },
        auto_startup_time: {
          label: 'Auto Run Trigger Time',
          hint: '24-hour format, HH:MM.',
        },
        battle_decision_backend: {
          label: 'Battle Decision Backend',
          hint: 'Decision method for card plays during Lesson/Exam stages.',
          option: {
            algo: {
              title: 'Simple Algorithm',
            },
            llm: {
              title: 'LLM (Beta)',
            },
            rl_battle: {
              title: 'RL (WIP)',
            },
          },
        },
        check_resource_updates_on_startup: {
          label: 'Check Resource Updates on Startup',
          hint: 'Run a resource repository update check immediately after every startup.',
        },
        disabled_tasks: {
          label: 'Disabled Task List',
          hint: 'Configure the list of disabled tasks.',
        },
        enable_game_asset_download: {
          label: 'Enable Game Asset Downloads',
          hint: 'Download game asset files from the game server with GkmasObjectManager. Internet access is required.',
        },
        enabled_auto_startup: {
          label: 'Run Script Automatically Every Day',
          hint: 'When enabled, the task queue starts automatically at the configured time.',
        },
        enabled_check_resource_updates: {
          label: 'Periodic Resource Update Checks',
          hint: 'Check assets/GakumasTranslationData and assets/gakumasu-diff for upstream updates on the configured schedule.',
        },
        gakumas_translation_data_repository_url: {
          label: 'GakumasTranslationData Repository URL',
          hint: 'Repository URL used for resource downloads and updates. Changing it immediately triggers a new update-state check. Leave it unchanged if you do not know what it is.',
        },
        gakumasu_diff_repository_url: {
          label: 'gakumasu-diff Repository URL',
          hint: 'Repository URL used for resource downloads and updates. Changing it immediately triggers a new update-state check. Leave it unchanged if you do not know what it is.',
        },
        game_package_name: {
          label: 'Game Package Name',
          hint: 'Default: com.bandainamcoent.idolmaster_gakuen. Restart required after changes.',
        },
        game_window_name: {
          label: 'Game Window Name',
          hint: 'Default: gakumas. Restart required after changes.',
        },
        llm_api_key: {
          label: 'LLM API Key',
          hint: 'API key.',
        },
        llm_base_url: {
          label: 'LLM API URL',
          hint: 'OpenAI-compatible API endpoint such as llama / vLLM / OpenAI.',
        },
        llm_insight_api_key: {
          label: 'Insight Model API Key',
          hint: 'Leave empty to use the primary LLM key.',
        },
        llm_insight_base_url: {
          label: 'Insight Model API URL',
          hint: 'Leave empty to use the primary LLM URL. You can point it to an independent model, such as a cloud model or a smaller local model.',
        },
        llm_insight_enabled: {
          label: 'Enable Strategy Insight',
          hint: 'Generate reusable strategy insights in the background for later decisions.',
        },
        llm_insight_max_tokens: {
          label: 'Insight Model Max Output Tokens',
          hint: '0 = unlimited. Let the model manage thinking and output token allocation on its own.',
        },
        llm_insight_model: {
          label: 'Insight Model',
          hint: 'Leave empty to use the primary LLM model name.',
        },
        llm_insight_num_ctx: {
          label: 'Insight Model Context Window',
          hint: '0 = unset. Let the API manage it automatically.',
        },
        llm_insight_reasoning_effort: {
          label: 'Insight Model Reasoning Effort',
          hint: 'Controls the reasoning depth used for insight generation.',
          option: {
            low: {
              title: 'low',
            },
            medium: {
              title: 'medium',
            },
            high: {
              title: 'high',
            },
            xhigh: {
              title: 'xhigh',
            },
          },
        },
        llm_insight_temperature: {
          label: 'Insight Model Temperature',
          hint: 'Sampling temperature for insight generation. Lower values are more deterministic (0.0 ~ 1.0).',
        },
        llm_insight_timeout: {
          label: 'Insight Model Timeout (s)',
          hint: 'Timeout for background insight generation. It can be longer than the main decision timeout.',
        },
        llm_max_tokens: {
          label: 'LLM Max Output Tokens',
          hint: 'Upper limit of output tokens, including thinking and final answer. Set to 0 for automatic mode (not sent to the API).',
        },
        llm_model: {
          label: 'LLM Model',
          hint: 'Model name, for example gpt-oss:20b, qwen3:4b, or qwen3.5:9b.',
        },
        llm_num_ctx: {
          label: 'LLM Context Window',
          hint: 'Optional compatibility parameter, mainly for Ollama or local OpenAI-compatible backends. Set to 0 for automatic mode.',
        },
        llm_reasoning_effort: {
          label: 'LLM Reasoning Effort',
          hint: 'Controls reasoning depth. low = faster, medium = balanced, high = more thorough.',
          option: {
            low: {
              title: 'low',
            },
            medium: {
              title: 'medium',
            },
            high: {
              title: 'high',
            },
            xhigh: {
              title: 'xhigh',
            },
          },
        },
        llm_temperature: {
          label: 'LLM Temperature',
          hint: 'Sampling temperature. Lower values are more deterministic (0.0 ~ 1.0).',
        },
        llm_timeout: {
          label: 'LLM Timeout (s)',
          hint: 'API request timeout.',
        },
        other_decision_backend: {
          label: 'Other Decision Backend',
          hint: 'Decision method used during dialogue, P-drink, skill rewards, consultation, item selection, and similar stages.',
          option: {
            algo: {
              title: 'Simple Algorithm',
            },
            llm: {
              title: 'LLM (Beta)',
            },
            rl_battle: {
              title: 'RL (WIP)',
            },
          },
        },
        playtools_port: {
          label: 'PlayTools Port',
          hint: 'Port number shown in the PlayCover window title bar as [localhost:port]. Restart required after changes.',
        },
        prefer_game_asset_image: {
          label: 'Always Prefer Game Asset Images in UI',
          hint: 'When enabled, item / support-card / skill-card images in the UI always prefer resources downloaded from the game server instead of screenshots captured during gameplay.',
        },
        resource_update_check_period: {
          label: 'Resource Update Check Interval',
          hint: 'Used only for scheduled checks.',
          option: {
            daily: {
              title: 'Every day',
            },
            every_3_days: {
              title: 'Every 3 days',
            },
            weekly: {
              title: 'Every week',
            },
          },
        },
        rl_inference_base_url: {
          label: 'RL Inference Service URL',
          hint: 'URL of the stateless RL inference service.',
        },
        rl_inference_timeout: {
          label: 'RL Inference Timeout (s)',
          hint: 'Timeout for requests sent to the RL service.',
        },
        schedule_decision_backend: {
          label: 'Schedule Decision Backend',
          hint: 'Automatic decision method used during weekly Schedule stages.',
          option: {
            algo: {
              title: 'Simple Algorithm',
            },
            llm: {
              title: 'LLM (Beta)',
            },
            rl_battle: {
              title: 'RL (WIP)',
            },
          },
        },
      },
      dmm_player: {
        game_exe_path: {
          label: 'Game Install Path',
          hint: 'Game installation path pointing to gakumas.exe. It is auto-detected by default, and normally does not need to be changed.',
        },
        viewer_id: {
          label: 'Viewer ID',
          hint: 'Auto-detected. No need to change unless necessary.',
        },
        open_id: {
          label: 'Open ID',
          hint: 'Auto-detected. No need to change unless necessary.',
        },
        pf_token: {
          label: 'PF Token',
          hint: 'Auto-detected. No need to change unless necessary.',
        },
      },
      task__auto_purchase: {
        weekly_gift: {
          label: 'Buy Weekly Gift Pack',
          hint: 'Check the gift-pack page daily for free purchasable items.',
        },
        daily_buy_list: {
          label: 'Daily Purchase Items',
          hint: 'Choose items in the exchange shop that are allowed for automatic purchase.',
        },
        refresh_shop: {
          label: 'Auto Refresh Exchange Shop',
          hint: 'Refresh the exchange shop automatically every day.',
        },
        use_gem_refresh: {
          label: 'Use Gems to Refresh Exchange Shop',
          hint: 'Continue refreshing with gems after the free refresh has been used.',
        },
      },
      task__auto_contest: {
        auto_reconfigure_team_before_challenge: {
          label: 'Auto Reconfigure Team Before Challenge',
          hint: 'Automatic configuration is still triggered if the team has empty slots.',
        },
        challenge_order: {
          label: 'Challenge Order',
          hint: 'The script looks for challenge targets according to the configured order.',
          option: {
            random: {
              title: 'Random',
            },
            highest_power: {
              title: 'Highest',
            },
            lowest_power: {
              title: 'Lowest',
            },
            balanced_power: {
              title: 'Middle',
            },
          },
        },
      },
      task__dispatch_work: {
        reconfigure_work_hours: {
          label: 'Reconfigure Dispatch Duration',
          hint: 'When enabled, the work duration is reset before dispatch starts.',
        },
        working_hours: {
          label: 'Dispatch Duration',
          hint: 'Only takes effect when "Reconfigure Dispatch Duration" is enabled.',
          option: {
            '4_h': {
              title: '4 hours (minimum)',
            },
            '8_h': {
              title: '8 hours',
            },
            '12_h': {
              title: '12 hours (maximum)',
            },
          },
        },
      },
      task__auto_enhancement_support_card: {
        enhance_r: {
          label: 'Enhance R Cards',
          hint: 'Automatically enhance support cards with R rarity.',
        },
        enhance_r_max_level: {
          label: 'Max R Enhancement Level',
          hint: 'Target maximum level for R cards.',
        },
        enhance_sr: {
          label: 'Enhance SR Cards',
          hint: 'Automatically enhance support cards with SR rarity.',
        },
        enhance_sr_max_level: {
          label: 'Max SR Enhancement Level',
          hint: 'Target maximum level for SR cards.',
        },
        enhance_ssr: {
          label: 'Enhance SSR Cards',
          hint: 'Automatically enhance support cards with SSR rarity.',
        },
        enhance_ssr_max_level: {
          label: 'Max SSR Enhancement Level',
          hint: 'Target maximum level for SSR cards.',
        },
        auto_limit_break: {
          label: 'Auto Limit Break',
          hint: 'Automatically perform a limit break when duplicate cards exist and the star cap has not been reached.',
        },
        auto_convert: {
          label: 'Auto Convert Overflowing Support Cards',
          hint: 'Automatically convert overflowing support cards into Support Certificates.',
        },
        whitelist_mode: {
          label: 'Whitelist Mode',
          hint: 'Enhance only cards selected in the whitelist.',
        },
        whitelist_card_ids: {
          label: 'Whitelisted Cards',
          hint: 'Choose support cards that are allowed to be enhanced automatically.',
        },
        whitelist: {
          no_selection: 'No whitelisted support card has been selected yet',
          open_dialog: 'Select Whitelisted Cards',
          dialog_title: 'Support Card Whitelist',
          search_placeholder: 'Search support card name (supports Chinese/Japanese/ID)',
          downloading_images: 'Downloading support-card related images...',
          add_to_whitelist: 'Add to Whitelist',
          limit_only: 'Limited',
          trigger_rate: 'Support Trigger Rate',
          filter: {
            rarity: 'Rarity',
            type: 'Type',
            plan: 'Plan',
          },
          rarity: {
            ssr: 'SSR',
            sr: 'SR',
            r: 'R',
            n: 'N',
          },
          type: {
            vocal: 'Vocal',
            dance: 'Dance',
            visual: 'Visual',
            assist: 'Assist',
            vocal_short: 'Vo',
            dance_short: 'Da',
            visual_short: 'Vi',
            stamina_short: 'Sta',
            assist_short: 'Sup',
          },
          plan: {
            plan1: 'Sense',
            plan2: 'Logic',
            plan3: 'Anomaly',
            common: 'Common',
          },
          card_category: {
            active_skill: 'Active Skill',
            mental_skill: 'Mental Skill',
            trouble: 'Trouble Card',
            free_skill: 'Free Skill',
            skill_card: 'Skill Card',
            p_item: 'P Item',
          },
          section: {
            support_ability: 'Support Ability',
            support_event: 'Support Event',
            attachments: 'Attached Rewards',
          },
        },
      },
      task__auto_producer: {
        scenario: {
          label: 'Scenario',
          hint: 'Select the produce scenario.',
          option: {
            hajime: {
              title: 'Hajime',
            },
            nia: {
              title: 'NIA (WIP)',
            },
          },
        },
        difficulty: {
          label: 'Difficulty',
          hint: 'Select the produce difficulty.',
          option: {
            regular: {
              title: 'Regular',
            },
            pro: {
              title: 'Pro',
            },
            master: {
              title: 'Master',
            },
          },
        },
        nia_difficulty: {
          label: 'NIA Difficulty',
          hint: 'Select the difficulty for the NIA scenario.',
          option: {
            pro: {
              title: 'Pro',
            },
            master: {
              title: 'Master',
            },
          },
        },
        target_idol_card_id: {
          label: 'Target Idol Card',
          hint: 'Target P-idol card ID. Leave empty to use the default selected card. Run "Refresh Idol Card Storage" first to learn card features.',
        },
        support_card_mode: {
          label: 'Support Card Setup',
          hint: 'Use automatic setup or a preset number.',
          option: {
            auto: {
              title: 'Auto Setup',
            },
            preset: {
              title: 'Preset Number',
            },
          },
        },
        support_card_preset_index: {
          label: 'Support Card Preset Number',
          hint: 'Which preset support-card setup to use.',
        },
        memory_mode: {
          label: 'Memory Setup',
          hint: 'Use automatic setup or a preset number.',
          option: {
            auto: {
              title: 'Auto Setup',
            },
            preset: {
              title: 'Preset Number',
            },
          },
        },
        memory_preset_index: {
          label: 'Memory Preset Number',
          hint: 'Which preset memory setup to use.',
        },
        use_rental: {
          label: 'Use Rental Memory',
          hint: 'When arranging memories automatically, check the "Use Rental" checkbox.',
        },
        use_boost_items: {
          label: 'Use Boost Items',
          hint: 'Whether to use boost items on the Start Confirmation page.',
        },
        resume_interrupted: {
          label: 'Resume Interrupted Produce',
          hint: 'If an interrupted produce session from last time is detected, resume it automatically instead of abandoning it and starting over.',
        },
        allow_ap_recovery: {
          label: 'Allow Item-based AP Recovery',
          hint: 'Whether AP can be restored automatically with items when AP is insufficient.',
        },
        allow_destroy_production_data: {
          label: 'Allow Destruction of Cross-device Incomplete Produce Data',
          hint: 'Whether to confirm and continue when a "Discard Produce Data" prompt appears.',
        },
        schedule_notebook_mode: {
          label: 'P Notebook Read Strategy',
          hint: 'disabled: never read; before_decision: read only before automatic decisions on weekly schedule actions.',
          option: {
            disabled: {
              title: 'Disabled',
            },
            before_decision: {
              title: 'Read Before Decision Only',
            },
          },
        },
        memory_photo_mode: {
          label: 'Memory Photo Selection',
          hint: 'How to choose the memory photo at the end of produce.',
          option: {
            first: {
              title: 'Always Choose the First',
            },
            vl: {
              title: 'Use VL visual model to choose the best photo',
            },
          },
        },
        memory_photo_vl_prompt: {
          label: 'VL Photo Selection Prompt',
          hint: 'Custom prompt used by the VL model when choosing a memory photo. Leave empty to use the default prompt.',
        },
        idol_card_browser: {
          no_selection: 'No target idol card has been selected yet',
          search_placeholder: 'Search idol card name (supports Chinese/Japanese/ID)',
          limit_only: 'Limited',
          total: 'Total',
          stamina: 'Stamina',
          after_training: 'After Training',
          rarity: {
            ssr: 'SSR',
            sr: 'SR',
            r: 'R',
          },
          plan: {
            plan1: 'Sense',
            plan2: 'Logic',
            plan3: 'Anomaly',
            common: 'Common',
          },
          attribute: {
            vocal: 'Vocal',
            dance: 'Dance',
            visual: 'Visual',
          },
          exam_effect: {
            parameter_buff: 'Stat Buff',
            review: 'Review',
            lesson_buff: 'Lesson Buff',
            concentration: 'Concentration',
            card_play_aggressive: 'Aggressive Play',
            full_power: 'Full Power',
          },
          filter: {
            rarity: 'Rarity',
            plan: 'Plan',
            attribute: 'Attribute',
            exam_effect: 'Exam Effect',
            character: 'Character',
          },
          section: {
            growth: 'Growth Stats',
            skill_card: 'Skill Card',
            item: 'Item',
          },
          skin: {
            before: 'Before Awakening',
            after: 'After Awakening',
          },
        },
      },
    },
    adb: {
      missing: 'adb is not installed. Please install Android SDK Platform-Tools and add adb to PATH.',
      noError: '',
      notConnected: 'No ADB device is currently connected.',
      invalidConnectMode: 'Invalid ADB connection mode: {mode}',
      deviceDisconnectedTarget: 'ADB device {target} is disconnected or not connected. Make sure the emulator or device is running.',
      deviceDisconnectedSerial: 'ADB device {serial} is disconnected or not connected. Make sure USB is connected, USB debugging is enabled, and refresh the device list in WebUI.',
      deviceDisconnected: 'ADB device is disconnected or not connected. Make sure the device is online and try again.',
      deviceOfflineTarget: 'ADB device {target} is currently offline. Retry or restart the device.',
      deviceOfflineSerial: 'ADB device {serial} is currently offline. Reconnect the device and confirm USB debugging authorization.',
      deviceOffline: 'ADB device is currently offline. Reconnect the device and try again.',
      usbNotFoundSerial: 'The selected USB ADB device was not found: {serial}. Make sure the device is connected, USB debugging is enabled, then refresh the device list in WebUI and select it again.',
      usbNotFound: 'No USB ADB device was detected. Connect a device, enable USB debugging, then refresh the device list in WebUI.',
      networkUnavailable: 'Failed to connect to ADB device {target}. Make sure adb is installed, the device is on the network, and adb tcpip / adb connect has been run. Original error: {message}',
      initFailed: 'ADB initialization failed: {message}',
    },
  },
  config: {
    loadingHint: 'If the form does not appear for a long time, restart the backend service to apply the updated configuration.',
    unavailableOptionsPrefix: 'Currently unavailable: {items}',
    optionDisabledItem: '{title}: {reason}',
    componentNotRegistered: 'Unregistered config component: {component}',
  },
  logger: {
    close: 'Close logs',
    open: 'Open logs',
    title: 'Execution Log',
  },
  api: {
    errorPrefix: 'API error: {message}{status}',
    statusSuffix: ' (status:{status})',
  },
})

export default en
