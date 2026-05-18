import { zhHant as vuetifyZhHant } from 'vuetify/locale'
import zhHans from './zhHans'
import { deepMergeMessages } from '@/scripts/i18n/messageTools'

/**
 * 繁體中文語言包。
 */
const zhHant = deepMergeMessages(zhHans, {
  $vuetify: vuetifyZhHant,
  app: {
    sections: {
      tasks: '任務列表',
      settings: '腳本設定',
      about: '關於專案',
    },
    footer: {
      exit: '退出應用',
      exiting: '應用正在退出...',
      teamName: 'Pigeon Server Team',
      licenseLabel: 'GPLv3 授權條款',
      closeLogger: '關閉日誌',
      openLogger: '打開日誌',
      executionLog: '執行日誌',
    },
    window: {
      minimize: '縮小',
      maximize: '放大',
      restore: '還原',
      close: '關閉',
    },
    preferences: {
      languageMenu: '切換語言',
      themeMenu: '切換主題',
    },
  },
  common: {
    confirm: '確認',
    cancel: '取消',
    save: '保存設定',
    reset: '恢復預設',
    loading: '載入中',
    notRun: '未執行',
    yes: '是',
    no: '否',
    unknown: '未知',
    close: '關閉',
    refresh: '重新整理',
    retry: '重試',
    auto: '自動',
    currentTask: '當前任務',
    manualOnly: '僅手動',
    system: '跟隨系統',
    light: '亮色',
    dark: '暗色',
  },
  dialogs: {
    resetSettingsTitle: '是否要重設所有設定項',
    resetSettingsText: '請謹慎操作，該操作會導致所有設定項恢復預設（包括任務設定）。',
    runFromTitle: '是否從這裡開始執行',
    runFromDescription: '將從「{task}」開始，按任務列表順序執行後續已啟用自動任務。',
    taskError: {
      title: '糟糕，任務執行失敗了',
      unknownTask: '未知任務',
      description: '因為某些原因，任務 {task} 執行失敗了，可以前往 GitHub Issues 或 QQ 群回報任務錯誤日誌，便於快速定位問題。',
      exception: '異常：{errorType} {errorMessage}',
      downloadLog: '下載日誌壓縮包',
      copyQqGroup: '複製 QQ 群號',
      openGithub: '打開 GitHub 回報',
      acknowledge: '知道了',
    },
    selector: {
      chooseCard: '選擇卡牌',
      selectCard: '選擇卡牌',
      cancelSelect: '取消選擇',
      searchPlaceholder: '搜尋卡牌名稱（支援中文/日文/ID）',
      selectedCount: '已選 {count}',
      totalCards: '{count} 張',
      noMatchedCards: '沒有符合的卡牌',
      clearSelection: '清空選擇',
      confirmSingle: '確認',
      confirmMulti: '確認 ({count})',
    },
  },
  websocket: {
    disconnected: '連線已中斷',
    reconnected: '伺服器重新連線成功',
    waitingServer: '等待伺服器回應.....',
    invalidBinaryFormat: '無效的格式',
  },
  toolbar: {
    deviceNotReady: '設備未就緒',
    deviceRetryHint: '{message} 點擊「開始執行」後會再次嘗試連接設備。',
    waitingAction: '等待中',
    running: '腳本執行中......',
    suspended: '腳本掛起中......',
    start: '開始執行',
    stop: '停止任務',
    suspend: '掛起任務',
    resume: '恢復任務',
    startQueued: '任務正在運行',
    stopQueued: '任務正在停止',
    suspendedDone: '任務已掛起',
    resumedDone: '任務已恢復',
  },
  settings: {
    title: '腳本設定',
    basicSection: '基礎設定',
    language: {
      label: '介面語言',
      hint: '切換 WebUI 顯示語言',
      option: {
        system: '跟隨系統',
        zhHans: '简体中文',
        zhHant: '繁體中文',
        en: 'English',
        ja: '日本語',
      },
    },
    theme: {
      label: '介面主題',
      option: {
        system: '跟隨系統',
        light: '亮色模式',
        dark: '暗色模式',
      },
    },
    saveSuccess: '設定保存成功',
    resetSuccess: '設定重設完成，部分設定可能需要重啟生效',
    refreshLaunchArgs: '刷新啟動參數',
    refreshLaunchArgsSuccess: '啟動參數刷新成功',
    resourceUpdate: {
      title: '資源更新',
      bootstrapAction: '下載所需資源',
      applyAction: '立刻更新',
      checkActionIdle: '檢查更新',
      checkActionBusy: '檢查中',
      state: {
        pending: '待檢查',
        checked: '已檢查',
        checking: '檢查中',
        downloading: '下載中',
        updating: '同步中',
        updateAvailable: '發現更新',
        bootstrapPending: '待下載',
        error: '檢查異常',
      },
      headline: {
        bootstrapIdle: '當前安裝包不再內建遊戲資料庫與本地化資源，首次啟動需要先下載',
        bootstrapRunning: '正在下載首次啟動所需的遊戲資料庫與本地化資源',
        updating: '正在同步資源倉庫並重新載入遊戲資料庫',
        checking: '正在檢查 GakumasTranslationData 與 gakumasu-diff 的上游更新',
        hasUpdate: '發現資源倉庫新版本，可以立即更新',
        lastError: '最近一次檢查存在異常',
        checked: '當前資源倉庫狀態已同步',
        idle: '可手動檢查，也可等待啟動或定時檢查',
      },
      bootstrapNotice: '首次啟動前需要先下載遊戲資料庫與本地化資源。下載失敗會自動重試，完成後程式會自動繼續初始化。',
      progressFallbackTitle: '正在處理資源',
      progress: {
        remoteHeadNotFound: '無法獲取遠端 HEAD',
        gitNotFound: '未找到 git 可執行檔案',
      },
      progressFallbackMessage: '正在同步資源，請稍候。',
      recentError: '最近錯誤：{error}',
      pendingDialogTitle: '首次啟動需要下載執行資源',
      pendingDialogDescription: '當前安裝包不再內建遊戲資料庫與本地化資源。確認後將自動下載，完成後程式會繼續初始化。',
      pendingDialogMeta: '首次啟動需要下載遊戲資料庫與本地化資源。',
      pendingDialogMetaItem: '{name}（缺少 {missing}/{required} 個檔案）',
      dialogAgree: '同意並開始下載',
      dialogLater: '稍後處理',
      downloadProgressTitle: '正在下載資源',
    },
  },
  tasks: {
    title: '任務列表',
    blockedHint: '資源下載完成前可先查看任務與設定，暫不可執行。',
    taskName: '任務名：',
    enabled: '啟用：',
    lastRunTime: '上次執行時間：',
    run: '執行',
    runFrom: '從這裡開始執行',
    disable: '禁用',
    enable: '啟用',
    settings: '任務設定',
    runBlockedTitle: '資源尚未準備完成，暫不可執行',
    runBusyTitle: '目前已有任務佇列在運行',
    runFromTitle: '從目前任務開始執行後續已啟用自動任務',
    status: {
      PENDING: '待執行',
      RUNNING: '運行中',
      SUSPENDED: '掛起中',
      SUCCESS: '完成',
      FAILED: '執行錯誤',
      CANCELED: '已中止',
      UNKNOWN: '未知狀態',
    },
    relativeTime: {
      justNow: '剛剛',
      secondsAgo: '{count} 秒前',
      minutesAgo: '{count}分鐘前',
      hoursAgo: '{count}小時前',
      yesterday: '昨日',
      daysAgo: '{count} 天前',
    },
  },
  resource: {
    progress: {
      step: '步驟 {current}/{total}',
      attempt: '嘗試 {current}/{total}',
      retryInSeconds: '{seconds}s 後重試',
      lastCheckedAt: '上次檢查：{time}',
      nextCheckAt: '下次定時檢查：{time}',
      bootstrapDownloadHint: '首次啟動需要下載遊戲資料庫與本地化資源',
      checkingHint: '正在檢查資源倉庫更新',
      updatingHint: '正在更新資源倉庫',
      notCheckedYet: '尚未執行資源倉庫檢查',
      checkFailed: '資源檢查失敗：{error}',
      checkHasUpdate: '檢測到資源倉庫更新，可立即更新',
      checkUpToDate: '資源倉庫已是最新版本',
      updateFailed: '資源更新失敗：{error}',
      bootstrapCompleted: '首次啟動所需資源下載完成，遊戲資料庫已重新載入',
      updateCompleted: '資源倉庫更新完成，遊戲資料庫已重新載入',
      bootstrapPromptSingle: '首次啟動需要下載遊戲資料庫與本地化資源。下載過程中會顯示進度，失敗會自動重試。是否現在開始下載？',
      bootstrapPromptMultiple: '首次啟動需要下載以下資源：{repositories}。下載過程中會顯示進度，失敗會自動重試。是否現在開始下載？',
      updatePromptSingle: '檢測到資源倉庫有更新，是否現在更新並重新載入遊戲資料庫？',
      updatePromptMultiple: '檢測到資源倉庫有更新：{repositories}。是否現在更新並重新載入遊戲資料庫？',
      repoMissingItem: '{name}（缺少 {missing}/{required} 個檔案）',
      repositoryCommitRange: '{name}（{local} -> {remote}）',
      updatePromptTitle: '發現資源倉庫更新',
      updatePromptConfirm: '立刻更新',
      updatePromptCancel: '稍後處理',
      checkCompleted: '資源倉庫檢查完成',
      bootstrapRunning: '正在下載首次啟動所需資源...',
      updateRunning: '正在更新資源倉庫...',
      checkCompletedWithErrors: '資源倉庫檢查完成，但部分倉庫檢查失敗：{error}',
      operationLocked: '目前正在檢查或更新資源倉庫',
      taskRunning: '任務執行中，無法更新資源倉庫',
      upToDate: '資源倉庫已是最新版本',
      reloadingDatabase: '正在重新載入遊戲資料庫',
      reloadingDatabaseMessage: '資源下載完成，正在重新載入遊戲資料庫與相關服務。',
      reloadFailed: '資源重新載入失敗：{error}',
      repositoryError: '{error}',
      noError: '',
      resourcesMissing: '資源下載尚未完成，仍缺少執行所需資源',
      retrying: '正在重試下載 {repository}',
      retryingMessage: '{repository} 下載失敗，將在 {seconds} 秒後自動重試。',
      retryExceeded: '已自動重試 {limit} 次，最後一次錯誤：{error}',
      updatingRepository: '正在更新 {repository}',
      updatingRepositoryWithGit: '正在透過 Git 更新 {repository}。',
      repositoryUpdated: '{repository} 已更新',
      repositorySynced: '{repository} 已同步至最新版本。',
      installingRepository: '正在安裝 {repository}',
      writingRepository: '正在將 {repository} 寫入本地資源目錄。',
      repositoryUpdatedToLatest: '{repository} 已更新到最新版本。',
      preparingRepositoryDownload: '正在準備下載 {repository}',
      preparingRepositoryDownloadMessage: '正在準備從 {url} 下載 {repository}。',
      downloadingRepository: '正在下載 {repository}',
      downloadingRepositoryFromGithub: '正在從 GitHub 下載 {repository} 資源包。',
      extractingRepository: '正在解壓 {repository}',
      extractingRepositoryMessage: '正在解壓 {repository} 資源包。',
      downloadingRepositoryWithGit: '正在透過 Git 下載 {repository} 資源包。',
    },
  },
  backend: {
    api: {
      ok: 'OK',
      genericError: 'error',
      invalidTaskName: '任務不存在',
      taskConfigMissing: '該任務沒有設定項。',
      shutdownStarted: '應用正在退出',
      resourceNotReady: '首次啟動需要先下載遊戲資料庫與本地化資源，請在 WebUI 中確認下載。',
      gameDatabaseNotReady: '遊戲資料庫資源尚未就緒',
      taskQueueStartFailed: '任務佇列啟動失敗',
      taskStartFailed: '任務啟動失敗',
      runFromFailed: '從目前任務開始執行失敗',
      manualOnlyRunFromUnsupported: '僅手動任務不支援從這裡開始執行',
      noRunningTask: '目前沒有正在運行的任務',
      noSuspendedTask: '目前沒有已掛起的任務',
      suspendUnsupported: '目前任務不支援手動掛起',
      resumeUnsupported: '目前任務不支援手動解除掛起',
      resumeBlockedByInsertedTask: '目前處於插隊執行中，無法恢復執行',
      taskFailurePackageMissing: '日誌壓縮包不存在或已失效，請重試任務後重新下載。',
      refreshDmmTokenFailed: '提取遊戲啟動參數失敗：{error}',
      imageDownloadDisabled: '遊戲資源下載功能未啟用，請在設定中開啟',
      imageDownloadFeatureDisabledShort: '遊戲資源下載功能未啟用',
      objectManagerUnavailable: 'GkmasObjectManager 未就緒，請確認 vendor/GkmasObjectManager 子模組已初始化',
      objectManagerUnavailableShort: 'GkmasObjectManager 未就緒',
      downloadInProgress: '正在下載中，請稍後',
      supportCardThumbDownloadStarted: '開始下載支援卡縮圖',
      supportCardFullDownloadStarted: '開始下載支援卡全尺寸圖片',
      downloadStarted: '開始下載',
      downloadAlreadyExists: '已存在於本地',
      downloadAlreadyRunning: '下載已在進行中',
      supportCardAutoDownloadStarted: '開始自動下載支援卡圖片',
      cardNotFound: '找不到卡牌：{cardId}',
    },
    app: {
      deviceInitializing: '正在初始化設備...',
      deviceUnavailable: '目前設備不可用。',
      deviceReadyAutoDetected: '已自動識別到可用設備',
      deviceDisconnected: '設備連接已斷開',
      status: {
        ready: '設備已就緒',
        initializing: '正在初始化設備',
      },
    },
    device: {
      windows: {
        available: '',
        unavailable: {
          non_windows: 'PC 模式僅支援 Windows，請在 macOS / Linux 上使用 Phone 模式。',
          import_error: 'PC 模式依賴的 Windows 專用元件未就緒（通常是 pywin32 未安裝或損壞），請重新執行 `pip install -r requirements.txt` 後重試。',
          unknown: 'PC 模式目前不可用。',
          init_failed: 'Windows 裝置初始化失敗：{message}',
        },
      },
      mac: {
        available: '',
        unavailable: {
          non_macos: 'MacPlayTools 模式僅支援 macOS（Apple Silicon）。',
          import_error: 'MacPlayTools 模式依賴的元件未就緒：{error}',
          unknown: 'MacPlayTools 模式目前不可用。',
          port_not_configured: 'MacPlayTools 端口未配置。請在 PlayCover 中啟動遊戲後，從視窗標題列取得 [localhost:埠號] 並填入設定。',
          connect_failed: '無法連接到 MacPlayTools ({host}:{port})。請確認 PlayCover 中的遊戲已啟動且 MaaTools 已啟用。',
          init_failed: 'MacPlayTools 裝置初始化失敗：{message}',
        },
      },
    },
    task: {
      startManual: '已開始手動執行任務：{task}',
      startFrom: '已從這裡開始執行後續任務：{task}',
      enabled: '已啟用任務：{task}',
      disabled: '已禁用任務：{task}',
      names: {
        start_game: '啟動遊戲',
        get_expenditure: '領取活動費',
        dispatch_work: '派遣工作',
        get_gift: '領取禮物 / 郵件',
        auto_purchase: '每日交換所自動購買',
        auto_enhancement_support_card: '自動強化支援卡',
        auto_contest: '每日競技場自動挑戰',
        claim_task_rewards: '領取任務獎勵',
        claim_pass_rewards: '領取通行證獎勵',
        auto_producer: '自動培育（Beta）',
        void_task: '測試任務',
        refresh_skill_storage: '刷新技能卡儲存',
        learn_support_card_clip: '刷新支援卡儲存',
        learn_idol_card_clip: '刷新偶像卡儲存',
      },
      gameNotInForeground: '遊戲未在前台運行，請手動切回遊戲後重試',
      gameNotStarted: '遊戲未啟動，請先手動啟動遊戲後重試',
      suspendSwitchToAlbum: '任務已掛起，請手動切換到圖鑑頁面',
      tabBarNotFound: '無法找到TabBar，刷新失敗',
      requiredButtonNotFound: '無法找到任務所需的按鈕，刷新失敗',
      cardAttributeSwitchNotFound: '無法找到卡屬性切換按鈕，刷新失敗',
      idolSwitchNotFound: '無法找到偶像切換按鈕，刷新失敗',
      startIdolCardCLIP: '開始偶像卡 CLIP 學習，將按底部卡條逐張遍歷',
      startSupportCardCLIP: '開始支援卡 CLIP 學習，請確保已進入支援卡列表頁面',
      noEnhancementConfigured: '未配置任何需要強化的品級',
      clipNotInitialized: 'CLIP 服務未初始化，無法學習偶像卡 CLIP',
      notOnIdolCardPage: '當前不在偶像卡養成頁面，未檢測到 PRODUCT_CARD_SELECTED',
      externalPageRecoveryFailed: '外頁恢復失敗：已回到主頁並打開 Produce，但未命中未完成培育再開彈窗',
      produceFinishTimeout: '培育收尾失敗：推進結果鏈超時仍未回到主頁',
      consecutiveUnknownsExceeded: '培育主循環: 連續無法識別畫面閾值超出',
      noHandlerPause: '培育主循環: 當前頁面無可用 handler，已暫停等待分析 (phase={phase}, position={position})',
      maxGameplayLoopsExceeded: '培育主循環: 達到最大循環次數 {max_loops}',
      supportCardSelectionFailed: '未能完成支援卡選擇',
      unknownSupportCardMode: '未知支援卡編成模式: {mode}',
      idolCardInfoNotFound: '無法獲取偶像卡資訊',
      trainingInfoCloseButtonNotFound: '未找到育成情報關閉按鈕，也無法推導蒙層關閉點',
      candidateListEmpty: '候選列表為空',
      combatPowerAnchorNotFound: '找不到[総合力合計]錨點',
    },
    gameAsset: {
      downloading: '正在下載 {label}...',
      fetchingManifest: '正在取得資源清單...',
      fetchingManifestWithLabel: '正在取得資源清單（{label}）...',
      searchingWithLabel: '正在檢索 {label} 資源...',
      noMatchingResources: '未找到 {label} 資源',
      noMatchingObjects: '未找到符合的物件',
      downloadCompleted: '{label} 下載完成：新增 {downloaded} 張圖片，略過 {skipped} 張（目前共快取 {total} 張）',
      phaseCompleted: '{label}：新增 {downloaded} 張圖片，略過 {skipped} 張（目前共快取 {total} 張）',
      downloadFailed: '{label} 下載失敗',
      downloadFailedError: '{error}',
      dialogAssetsCompleted: '支援卡相關資源已全部下載完成',
      bulkDownloadFailed: '資源下載失敗',
    },
    message: {
      gameNotForegroundRetry: '遊戲未在前台運行，請手動切回遊戲後重試',
      gameNotStartedRetry: '遊戲未啟動，請先手動啟動遊戲後重試',
      gameNotForegroundStart: '遊戲未在前台運行，請手動啟動遊戲後重試',
      startLearningIdolClip: '開始偶像卡 CLIP 學習，將按底部卡條逐張遍歷',
    },
    config: {
      section: {
        base: '基礎設定',
        dmm_player: 'DMM Player',
      },
      base: {
        run_mode: {
          label: '運行模式',
          hint: '腳本的執行模式（需重啟生效）',
          option: {
            pc: '電腦端（DMM）',
            phone: '手機端',
            mac_play_tools: 'macOS PlayCover',
          },
          disabledReason: {
            pcWindowsOnly: 'PC / DMM 模式僅支援 Windows。',
            macOnly: 'MacPlayTools 模式僅支援 macOS (Apple Silicon)。',
          },
        },
        ocr_backend: {
          label: 'OCR 後端',
          hint: 'auto：macOS 優先 Vision，其他平台使用 RapidOCR；失敗時會回退到 RapidOCR（修改後需重啟生效）',
          option: {
            auto: '自動',
            rapidocr: 'RapidOCR',
            vision: 'Vision（macOS 原生 OCR）',
          },
          disabledReason: {
            visionMacOnly: 'Vision OCR 僅在 macOS 可用。',
          },
        },
        adb_connect_mode: {
          label: 'ADB 連接模式',
          hint: 'Android 偵錯橋的連接模式，手機建議使用 USB，模擬器可使用網路連接（修改後需重啟生效）',
          option: {
            network: {
              title: '網路連接',
            },
            u_s_b: {
              title: 'USB 連接',
            },
          },
        },
        adb_host: {
          label: 'ADB 主機名稱',
          hint: 'Android 偵錯橋的 IP 位址，模擬器通常為 127.0.0.1',
        },
        adb_port: {
          label: 'ADB 連接埠',
          hint: 'Android 偵錯橋的連接埠，預設為 5555，Android 11 以上可能為系統隨機分配',
        },
        adb_serial: {
          label: '透過 USB 連接的 ADB 裝置',
          hint: '請選擇透過 USB 連接的裝置，如未找到請嘗試重新整理列表',
        },
        android_screen_capture_service: {
          label: 'ADB 截圖方式',
          hint: 'scrcpy / DroidCast 的延遲通常優於 ADB 截圖；使用 scrcpy 需將官方 Releases 的 scrcpy-server 放到 bin',
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
          label: 'ADB 點擊方式',
          hint: '可選 MaaTouch / minitouch / scrcpy；MaaTouch 需放入官方構建產物到 bin/maatouch 或用 workflow 生成，minitouch 需放入官方構建產物到 bin/minitouch，且目前僅支援 Android 9 及以下',
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
          label: '自動啟動遊戲',
          hint: '當遊戲尚未啟動時是否自動啟動遊戲',
        },
        auto_startup_time: {
          label: '自動執行觸發時間',
          hint: '24 小時制，格式為 HH:MM',
        },
        battle_decision_backend: {
          label: '戰鬥決策後端',
          hint: 'レッスン / 試験（Lesson / Exam）階段的出牌決策方式',
          option: {
            algo: {
              title: '簡單演算法決策',
            },
            llm: {
              title: 'LLM（Beta）',
            },
            rl_battle: {
              title: 'RL（WIP）',
            },
          },
        },
        check_resource_updates_on_startup: {
          label: '啟動時檢查資源倉庫更新',
          hint: '每次啟動後立即檢查一次資源倉庫更新',
        },
        disabled_tasks: {
          label: '禁用任務列表',
          hint: '設定要禁用的任務列表',
        },
        enable_game_asset_download: {
          label: '啟用遊戲資源下載',
          hint: '使用 GkmasObjectManager 從遊戲伺服器下載遊戲資源檔案，需要網際網路連線。',
        },
        enabled_auto_startup: {
          label: '每日自動執行腳本',
          hint: '啟用後會在設定時間自動開始執行任務佇列',
        },
        enabled_check_resource_updates: {
          label: '定時檢查資源倉庫更新',
          hint: '依設定週期檢查 assets/GakumasTranslationData 與 assets/gakumasu-diff 是否有上游更新',
        },
        gakumas_translation_data_repository_url: {
          label: 'GakumasTranslationData 倉庫 URL',
          hint: '資源下載與更新使用的倉庫位址。修改後會立即重新檢查更新狀態。（如果你不知道這是什麼，請不要更改）',
        },
        gakumasu_diff_repository_url: {
          label: 'gakumasu-diff 倉庫 URL',
          hint: '資源下載與更新使用的倉庫位址。修改後會立即重新檢查更新狀態。（如果你不知道這是什麼，請不要更改）',
        },
        game_package_name: {
          label: '遊戲套件名稱',
          hint: '預設：com.bandainamcoent.idolmaster_gakuen（修改後需重啟生效）',
        },
        game_window_name: {
          label: '遊戲視窗名稱',
          hint: '預設：gakumas（修改後需重啟生效）',
        },
        llm_api_key: {
          label: 'LLM API Key',
          hint: 'API 金鑰',
        },
        llm_base_url: {
          label: 'LLM API 位址',
          hint: 'OpenAI 相容 API 端點（llama / vLLM / OpenAI 等）',
        },
        llm_insight_api_key: {
          label: '洞察模型 API Key',
          hint: '留空則使用主 LLM Key',
        },
        llm_insight_base_url: {
          label: '洞察模型 API 位址',
          hint: '留空則使用主 LLM 位址；可配置獨立模型（雲端 / 小模型）',
        },
        llm_insight_enabled: {
          label: '啟用策略洞察',
          hint: '在背景生成可遷移的策略洞察，供後續決策參考',
        },
        llm_insight_max_tokens: {
          label: '洞察模型最大輸出 Token',
          hint: '0 = 不限制，讓模型自行管理 thinking + output 的 token 分配',
        },
        llm_insight_model: {
          label: '洞察模型',
          hint: '留空則使用主 LLM 模型名稱',
        },
        llm_insight_num_ctx: {
          label: '洞察模型上下文視窗',
          hint: '0 = 不設定，由 API 自動管理',
        },
        llm_insight_reasoning_effort: {
          label: '洞察模型思考強度',
          hint: '控制洞察生成的推理深度',
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
          label: '洞察模型溫度',
          hint: '洞察生成的溫度，越低越確定（0.0 ~ 1.0）',
        },
        llm_insight_timeout: {
          label: '洞察模型逾時（秒）',
          hint: '背景洞察生成的逾時時間，可比主決策更長',
        },
        llm_max_tokens: {
          label: 'LLM 最大輸出 Token',
          hint: '輸出 token 上限（包含思考 + 回答），設為 0 表示自動（不傳給 API）',
        },
        llm_model: {
          label: 'LLM 模型',
          hint: '模型名稱（例如 gpt-oss:20b、qwen3:4b、qwen3.5:9b 等）',
        },
        llm_num_ctx: {
          label: 'LLM 上下文視窗',
          hint: '可選相容參數，主要供 Ollama / 本地 OpenAI 相容後端覆蓋上下文視窗；設為 0 表示自動',
        },
        llm_reasoning_effort: {
          label: 'LLM 思考強度',
          hint: '控制推理深度；low = 更快，medium = 平衡，high = 更充分',
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
          label: 'LLM 溫度',
          hint: '生成溫度，越低越確定（0.0 ~ 1.0）',
        },
        llm_timeout: {
          label: 'LLM 逾時（秒）',
          hint: 'API 請求逾時時間',
        },
        other_decision_backend: {
          label: '其他決策後端',
          hint: '對話 / P 飲料 / 技能獎勵 / 諮詢 / 道具選擇等階段的決策方式',
          option: {
            algo: {
              title: '簡單演算法決策',
            },
            llm: {
              title: 'LLM（Beta）',
            },
            rl_battle: {
              title: 'RL（WIP）',
            },
          },
        },
        playtools_port: {
          label: 'PlayTools 連接埠',
          hint: 'PlayCover 遊戲視窗標題列 [localhost:連接埠號] 中的連接埠號（修改後需重啟生效）',
        },
        prefer_game_asset_image: {
          label: '在 UI 中始終使用遊戲資源圖片',
          hint: '啟用後，UI 中的物品 / 支援卡 / 技能卡圖片會優先使用從遊戲伺服器下載的資源圖片，而不是遊戲過程中截圖識別到的圖像。',
        },
        resource_update_check_period: {
          label: '資源倉庫檢查週期',
          hint: '僅用於定時檢查',
          option: {
            daily: {
              title: '每天',
            },
            every_3_days: {
              title: '每 3 天',
            },
            weekly: {
              title: '每週',
            },
          },
        },
        rl_inference_base_url: {
          label: 'RL 推理服務位址',
          hint: '無狀態 RL 推理服務位址',
        },
        rl_inference_timeout: {
          label: 'RL 推理逾時（秒）',
          hint: '請求 RL 服務的逾時時間',
        },
        schedule_decision_backend: {
          label: '週行程決策後端',
          hint: '週行程（Schedule）階段的自動決策方式',
          option: {
            algo: {
              title: '簡單演算法決策',
            },
            llm: {
              title: 'LLM（Beta）',
            },
            rl_battle: {
              title: 'RL（WIP）',
            },
          },
        },
      },
      dmm_player: {
        game_exe_path: {
          label: '遊戲安裝目錄',
          hint: '遊戲安裝路徑，指向 gakumas.exe（預設自動取得，非必要無需修改）',
        },
        viewer_id: {
          label: 'Viewer ID',
          hint: '自動取得，非必要無需修改',
        },
        open_id: {
          label: 'Open ID',
          hint: '自動取得，非必要無需修改',
        },
        pf_token: {
          label: 'PF Token',
          hint: '自動取得，非必要無需修改',
        },
      },
      task__auto_purchase: {
        weekly_gift: {
          label: '購買每週禮包',
          hint: '每日檢查禮包頁面是否有可免費購買的項目',
        },
        daily_buy_list: {
          label: '每日購買物品',
          hint: '從交換所中選擇允許自動購買的物品',
        },
        refresh_shop: {
          label: '自動刷新交換所',
          hint: '每日自動刷新交換所',
        },
        use_gem_refresh: {
          label: '使用鑽石刷新交換所',
          hint: '免費刷新後仍可使用鑽石繼續刷新',
        },
      },
      task__auto_contest: {
        auto_reconfigure_team_before_challenge: {
          label: '挑戰前自動重新配置隊伍',
          hint: '若隊伍中有空位，仍會觸發自動配置',
        },
        challenge_order: {
          label: '挑戰順序',
          hint: '腳本會依設定順序尋找符合條件的挑戰對象',
          option: {
            random: {
              title: '隨機選擇',
            },
            highest_power: {
              title: '最高',
            },
            lowest_power: {
              title: '最低',
            },
            balanced_power: {
              title: '中間',
            },
          },
        },
      },
      task__dispatch_work: {
        reconfigure_work_hours: {
          label: '重新設定任務派遣時間',
          hint: '開啟後會在派遣前重新設定工作時長',
        },
        working_hours: {
          label: '任務派遣時間',
          hint: '僅在開啟「重新設定任務派遣時間」時生效',
          option: {
            '4_h': {
              title: '4 小時（最低）',
            },
            '8_h': {
              title: '8 小時',
            },
            '12_h': {
              title: '12 小時（最高）',
            },
          },
        },
      },
      task__auto_enhancement_support_card: {
        enhance_r: {
          label: '強化 R 卡',
          hint: '自動強化 R 稀有度的支援卡',
        },
        enhance_r_max_level: {
          label: 'R 最大強化等級',
          hint: 'R 卡的最大目標等級',
        },
        enhance_sr: {
          label: '強化 SR 卡',
          hint: '自動強化 SR 稀有度的支援卡',
        },
        enhance_sr_max_level: {
          label: 'SR 最大強化等級',
          hint: 'SR 卡的最大目標等級',
        },
        enhance_ssr: {
          label: '強化 SSR 卡',
          hint: '自動強化 SSR 稀有度的支援卡',
        },
        enhance_ssr_max_level: {
          label: 'SSR 最大強化等級',
          hint: 'SSR 卡的最大目標等級',
        },
        auto_limit_break: {
          label: '自動進行上限突破',
          hint: '有同名卡且未達星級上限時，自動進行上限突破',
        },
        auto_convert: {
          label: '自動轉換溢出的支援卡',
          hint: '自動將溢出的支援卡轉換為「サポートの証」',
        },
        whitelist_mode: {
          label: '白名單模式',
          hint: '僅強化白名單中選擇的卡牌',
        },
        whitelist_card_ids: {
          label: '白名單卡牌',
          hint: '選擇允許自動強化的支援卡',
        },
        whitelist: {
          no_selection: '尚未選擇白名單支援卡',
          open_dialog: '選擇白名單卡牌',
          dialog_title: '支援卡白名單',
          search_placeholder: '搜尋支援卡名稱（支援中文 / 日文 / ID）',
          downloading_images: '正在下載支援卡相關圖片...',
          add_to_whitelist: '加入白名單',
          limit_only: '限定',
          trigger_rate: '支援發生率',
          filter: {
            rarity: '稀有度',
            type: '類型',
            plan: '路線',
          },
          type: {
            stamina_short: '體',
            assist_short: '輔',
          },
          plan: {
            plan1: '感性',
            plan2: '邏輯',
            plan3: '異常',
            common: '通用',
          },
          card_category: {
            active_skill: '主動技能',
            mental_skill: '精神技能',
            trouble: '麻煩卡',
            free_skill: '自由技能',
            skill_card: '技能卡',
            p_item: 'P 物品',
          },
          section: {
            support_ability: '支援能力',
            support_event: '支援事件',
            attachments: '附帶獎勵',
          },
        },
      },
      task__auto_producer: {
        scenario: {
          label: '劇本',
          hint: '選擇培育劇本',
          option: {
            hajime: {
              title: '初',
            },
          },
        },
        difficulty: {
          label: '難度',
          hint: '選擇培育難度',
        },
        nia_difficulty: {
          label: 'NIA 難度',
          hint: '選擇 NIA 劇本難度',
        },
        target_idol_card_id: {
          label: '目標偶像卡',
          hint: '目標 P アイドル ID（留空則使用預設選中的卡；需先執行「刷新偶像卡儲存」學習卡片特徵）',
        },
        support_card_mode: {
          label: '支援卡編成',
          hint: '自動編成或使用預設編號',
          option: {
            auto: {
              title: '自動編成',
            },
            preset: {
              title: '預設編號',
            },
          },
        },
        support_card_preset_index: {
          label: '支援卡預設編號',
          hint: '使用第幾組預設編成',
        },
        memory_mode: {
          label: '記憶編成',
          hint: '自動編成或使用預設編號',
          option: {
            auto: {
              title: '自動編成',
            },
            preset: {
              title: '預設編號',
            },
          },
        },
        memory_preset_index: {
          label: '記憶預設編號',
          hint: '使用第幾組預設編成',
        },
        use_rental: {
          label: '使用租借記憶',
          hint: '自動安排記憶時勾選「レンタルを使用」核取方塊',
        },
        use_boost_items: {
          label: '使用加成道具',
          hint: '「開始確認」頁面是否使用加成道具（位於編成詳細按鈕上方）',
        },
        resume_interrupted: {
          label: '恢復中斷培育',
          hint: '偵測到上次中斷的培育時自動恢復（點擊「再開する」），而非放棄後重新開始',
        },
        allow_ap_recovery: {
          label: '允許使用道具恢復 AP',
          hint: 'AP 不足時是否允許自動消耗道具恢復',
        },
        allow_destroy_production_data: {
          label: '允許銷毀跨裝置未完成培育會話',
          hint: '偵測到「プロデュースデータの破棄」時是否允許確認並繼續',
        },
        schedule_notebook_mode: {
          label: 'P 手帳讀取策略',
          hint: 'disabled：不讀取；before_decision：僅在週行動自動決策前讀取（決策後不再觸發）',
          option: {
            disabled: {
              title: '關閉讀取',
            },
            before_decision: {
              title: '僅決策前讀取',
            },
          },
        },
        memory_photo_mode: {
          label: '記憶卡面選擇',
          hint: '培育結束後選擇記憶卡面的方式',
          option: {
            first: {
              title: '預設選第一張',
            },
            vl: {
              title: '由 VL 視覺模型自動選擇最佳卡面',
            },
          },
        },
        memory_photo_vl_prompt: {
          label: 'VL 選卡面提示詞',
          hint: '自訂 VL 模型選擇卡面時使用的提示詞（留空使用預設）',
        },
        idol_card_browser: {
          no_selection: '尚未選擇目標偶像卡',
          search_placeholder: '搜尋偶像卡名稱（支援中文 / 日文 / ID）',
          limit_only: '限定',
          total: '總計',
          stamina: '體力',
          after_training: '培育後',
          plan: {
            plan1: '感性',
            plan2: '邏輯',
            plan3: '異常',
            common: '通用',
          },
          attribute: {
            vocal: '聲樂',
            dance: '舞蹈',
            visual: '表現',
          },
          exam_effect: {
            parameter_buff: '參數提升',
            review: '復習',
            lesson_buff: '課程加成',
            concentration: '專注',
            card_play_aggressive: '攻擊性出牌',
            full_power: '全力',
          },
          filter: {
            rarity: '稀有度',
            plan: '路線',
            attribute: '屬性',
            exam_effect: '考試效果',
            character: '角色',
          },
          section: {
            growth: '成長屬性',
            skill_card: '技能卡',
            item: '道具',
          },
          skin: {
            before: '覺醒前',
            after: '覺醒後',
          },
        },
      },
    },
    adb: {
      missing: '未安裝 adb，請先安裝 Android SDK Platform-Tools 並將 adb 加入 PATH。',
      noError: '',
      notConnected: '目前未連接到 ADB 裝置。',
      invalidConnectMode: '無效的 ADB 連接模式：{mode}',
      deviceDisconnectedTarget: 'ADB 裝置 {target} 已斷開或未連接。請確認模擬器 / 裝置正在運行。',
      deviceDisconnectedSerial: 'ADB 裝置 {serial} 已斷開或未連接。請確認 USB 已連接、已開啟 USB 偵錯，並在 WebUI 中重新整理裝置列表。',
      deviceDisconnected: 'ADB 裝置已斷開或未連接。請確認裝置在線後重試。',
      deviceOfflineTarget: 'ADB 裝置 {target} 目前處於離線狀態。請重試或重新啟動裝置。',
      deviceOfflineSerial: 'ADB 裝置 {serial} 目前處於離線狀態。請重新插拔裝置並確認 USB 偵錯授權。',
      deviceOffline: 'ADB 裝置目前處於離線狀態。請重新連接裝置後重試。',
      usbNotFoundSerial: '未找到所選 USB ADB 裝置：{serial}。請確認裝置已連接、已開啟 USB 偵錯，並在 WebUI 中重新整理裝置列表後再重新選擇。',
      usbNotFound: '未偵測到可用的 USB ADB 裝置。請連接裝置、開啟 USB 偵錯，並在 WebUI 中重新整理裝置列表。',
      networkUnavailable: '無法連接到 ADB 裝置 {target}。請確認 adb 已安裝、裝置已聯網且已執行 adb tcpip / adb connect，原始錯誤：{message}',
      initFailed: 'ADB 初始化失敗：{message}',
      unavailable: 'Android 裝置不可用。',
      touchServiceUnavailable: '點擊失敗：觸控服務不可用',
      reconnectFailed: '重新連接 ADB 裝置失敗',
    },
    gameUtils: {
      modalHeaderTopEmpty: '無法讀取模態標題位置：modal 為空',
      modalMissingHeaderBox: '模態 \'{title}\' 缺少 header_box，無法判斷穩定性',
      modalHeaderMissingCoords: '模態 \'{title}\' 的 header_box 缺少 y/cy 坐標，無法判斷穩定性',
      modalSignatureEmpty: '無法構建模態簽名：modal 為空',
      modalSignatureMissingHeader: '模態 \'{title}\' 缺少 header_box，無法構建簽名',
      modalHeaderNoCenter: '模態 \'{title}\' 的 header_box 無法讀取中心點',
      modalNoActionButtons: '模態 \'{title}\' 缺少可用的操作按鈕，無法構建簽名',
    },
  },
  config: {
    loadingHint: '若長時間未顯示，請重啟後端服務以套用設定更新。',
    unavailableOptionsPrefix: '目前不可選：{items}',
    optionDisabledItem: '{title}：{reason}',
    componentNotRegistered: '未註冊的配置元件：{component}',
  },
  logger: {
    close: '關閉日誌',
    open: '打開日誌',
    title: '執行日誌',
  },
  api: {
    errorPrefix: 'API 錯誤：{message}{status}',
  },
})

export default zhHant
