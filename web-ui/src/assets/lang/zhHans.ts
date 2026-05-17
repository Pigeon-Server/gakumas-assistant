import { zhHans as vuetifyZhHans } from 'vuetify/locale'

/**
 * 简体中文语言包。
 */
const zhHans = {
  $vuetify: vuetifyZhHans,
  app: {
    name: 'Gakumas Assistant',
    sections: {
      tasks: '任务列表',
      settings: '脚本配置',
      about: '关于项目',
    },
    footer: {
      exit: '退出应用',
      exiting: '应用正在退出...',
      teamName: 'Pigeon Server Team',
      licenseLabel: 'GPLv3 许可证',
      closeLogger: '关闭日志',
      openLogger: '打开日志',
      executionLog: '执行日志',
    },
    window: {
      minimize: '最小化',
      maximize: '最大化',
      restore: '还原',
      close: '关闭',
    },
    preferences: {
      languageMenu: '切换语言',
      themeMenu: '切换主题',
    },
  },
  common: {
    confirm: '确认',
    cancel: '取消',
    save: '保存设置',
    reset: '恢复默认',
    loading: '加载中',
    notRun: '未运行',
    yes: '是',
    no: '否',
    unknown: '未知',
    close: '关闭',
    refresh: '刷新',
    retry: '重试',
    auto: '自动',
    currentTask: '当前任务',
    manualOnly: '仅手动',
    system: '跟随系统',
    light: '亮色',
    dark: '暗色',
  },
  dialogs: {
    resetSettingsTitle: '是否要重置所有设置项',
    resetSettingsText: '请谨慎操作，该操作会导致所有设置项恢复默认（包括任务设置）。',
    runFromTitle: '是否从这里开始执行',
    runFromDescription: '将从“{task}”开始，按任务列表顺序执行后续已启用自动任务。',
    taskError: {
      title: '糟糕，任务执行失败了',
      unknownTask: '未知任务',
      description: '因为某些原因，任务 {task} 执行失败了，可以前往 GitHub Issues 或 QQ 群反馈任务错误日志，便于快速定位问题。',
      exception: '异常：{errorType} {errorMessage}',
      downloadLog: '下载日志压缩包',
      copyQqGroup: '复制 QQ 群号',
      openGithub: '打开 GitHub 反馈',
      acknowledge: '我知道了',
    },
    selector: {
      chooseCard: '选择卡牌',
      selectCard: '选择卡牌',
      cancelSelect: '取消选择',
      searchPlaceholder: '搜索卡牌名称（支持中文/日文/ID）',
      selectedCount: '已选 {count}',
      totalCards: '{count} 张',
      noMatchedCards: '没有匹配的卡牌',
      clearSelection: '清空选择',
      confirmSingle: '确认',
      confirmMulti: '确认 ({count})',
    },
  },
  websocket: {
    disconnected: '连接已断开',
    reconnected: '服务器重连成功',
    waitingServer: '等待服务器响应.....',
    invalidBinaryFormat: '无效的格式',
  },
  toolbar: {
    deviceNotReady: '设备未就绪',
    deviceRetryHint: '{message} 点击“开始执行”后会再次尝试连接设备。',
    waitingAction: '等待操作',
    running: '脚本执行中......',
    suspended: '脚本挂起中......',
    start: '开始执行',
    stop: '停止任务',
    suspend: '挂起任务',
    resume: '恢复任务',
    startQueued: '任务正在运行',
    stopQueued: '任务正在停止',
    suspendedDone: '任务已挂起',
    resumedDone: '任务已恢复',
  },
  settings: {
    title: '脚本设置',
    basicSection: '基础设置',
    language: {
      label: '界面语言',
      hint: '切换 WebUI 显示语言',
      option: {
        system: '跟随系统',
        zhHans: '简体中文',
        zhHant: '繁體中文',
        en: 'English',
        ja: '日本語',
      },
    },
    theme: {
      label: '界面主题',
      option: {
        system: '跟随系统',
        light: '亮色模式',
        dark: '暗色模式',
      },
    },
    saveSuccess: '设置保存成功',
    resetSuccess: '设置重置完成，部分设置可能需要重启生效',
    refreshLaunchArgs: '刷新启动参数',
    refreshLaunchArgsSuccess: '启动参数刷新成功',
    resourceUpdate: {
      title: '资源更新',
      bootstrapAction: '下载所需资源',
      applyAction: '立即更新',
      checkActionIdle: '检查更新',
      checkActionBusy: '检查中',
      state: {
        pending: '待检查',
        checked: '已检查',
        checking: '检查中',
        downloading: '下载中',
        updating: '更新中',
        updateAvailable: '发现更新',
        bootstrapPending: '待下载',
        error: '检查异常',
      },
      headline: {
        bootstrapIdle: '当前安装包不再内置游戏数据库和本地化资源，首次启动需要先下载',
        bootstrapRunning: '正在下载首次启动所需的游戏数据库和本地化资源',
        updating: '正在同步资源仓库并重新加载游戏数据库',
        checking: '正在检查 GakumasTranslationData 和 gakumasu-diff 的上游更新',
        hasUpdate: '发现资源仓库新版本，可以立即更新',
        lastError: '最近一次检查存在异常',
        checked: '当前资源仓库状态已同步',
        idle: '可手动检查，也可等待启动或定时检查',
      },
      bootstrapNotice: '首次启动前需要先下载游戏数据库和本地化资源。下载失败会自动重试，完成后程序会自动继续初始化。',
      progressFallbackTitle: '正在处理资源',
      progressFallbackMessage: '正在同步资源，请稍候。',
      recentError: '最近错误：{error}',
      pendingDialogTitle: '首次启动需要下载运行资源',
      pendingDialogDescription: '当前安装包不再内置游戏数据库和本地化资源。确认后将自动下载，完成后程序会继续初始化。',
      pendingDialogMeta: '首次启动需要下载游戏数据库和本地化资源。',
      pendingDialogMetaItem: '{name}（缺少 {missing}/{required} 个文件）',
      dialogAgree: '同意并开始下载',
      dialogLater: '稍后处理',
      downloadProgressTitle: '正在下载资源',
    },
  },
  tasks: {
    title: '任务列表',
    blockedHint: '资源下载完成前可先查看任务和配置，暂不可执行。',
    taskName: '任务名：',
    enabled: '启用：',
    lastRunTime: '上次运行时间：',
    run: '执行',
    runFrom: '从这里开始执行',
    disable: '禁用',
    enable: '启用',
    settings: '任务设置',
    runBlockedTitle: '资源未准备完成，暂不可执行',
    runBusyTitle: '当前已有任务队列在运行',
    runFromTitle: '从当前任务开始执行后续已启用自动任务',
    status: {
      PENDING: '等待中',
      RUNNING: '运行中',
      SUSPENDED: '挂起中',
      SUCCESS: '已完成',
      FAILED: '执行错误',
      CANCELED: '已取消',
      UNKNOWN: '未知状态',
    },
    relativeTime: {
      justNow: '刚刚',
      secondsAgo: '{count}秒前',
      minutesAgo: '{count}分钟前',
      hoursAgo: '{count}小时前',
      yesterday: '昨天',
      daysAgo: '{count}天前',
    },
  },
  resource: {
    bytes: {
      B: 'B',
      KB: 'KB',
      MB: 'MB',
      GB: 'GB',
    },
    progress: {
      step: '步骤 {current}/{total}',
      attempt: '尝试 {current}/{total}',
      retryInSeconds: '{seconds}s 后重试',
      lastCheckedAt: '上次检查：{time}',
      nextCheckAt: '下次定时检查：{time}',
      bootstrapDownloadHint: '首次启动需要下载游戏数据库和本地化资源',
      checkingHint: '正在检查资源仓库更新',
      updatingHint: '正在更新资源仓库',
      notCheckedYet: '尚未执行资源仓库检查',
      checkFailed: '资源检查失败：{error}',
      checkHasUpdate: '检测到资源仓库更新，可立即更新',
      checkUpToDate: '资源仓库已经是最新版本',
      updateFailed: '资源更新失败：{error}',
      bootstrapCompleted: '首次启动所需资源下载完成，游戏数据库已重新加载',
      updateCompleted: '资源仓库更新完成，游戏数据库已重新加载',
      bootstrapPromptSingle: '首次启动需要下载游戏数据库和本地化资源。下载过程中会显示进度，失败会自动重试。是否现在开始下载？',
      bootstrapPromptMultiple: '首次启动需要下载以下资源：{repositories}。下载过程中会显示进度，失败会自动重试。是否现在开始下载？',
      updatePromptSingle: '检测到资源仓库有更新，是否现在更新并重新加载游戏数据库？',
      updatePromptMultiple: '检测到资源仓库有更新：{repositories}。是否现在更新并重新加载游戏数据库？',
      repoMissingItem: '{name}（缺少 {missing}/{required} 个文件）',
      repositoryCommitRange: '{name}（{local} -> {remote}）',
      updatePromptTitle: '发现资源仓库更新',
      updatePromptConfirm: '立即更新',
      updatePromptCancel: '稍后处理',
      checkCompleted: '资源仓库检查完成',
      checkCompletedWithErrors: '资源仓库检查完成，但部分仓库检查失败：{error}',
      bootstrapRunning: '正在下载首次启动所需资源...',
      updateRunning: '正在更新资源仓库...',
      operationLocked: '当前正在检查或更新资源仓库',
      taskRunning: '任务执行中，无法更新资源仓库',
      upToDate: '资源仓库已经是最新版本',
      reloadingDatabase: '正在重载游戏数据库',
      reloadingDatabaseMessage: '资源下载完成，正在重载游戏数据库和相关服务。',
      reloadFailed: '资源重载失败：{error}',
      repositoryError: '{error}',
      noError: '',
      resourcesMissing: '资源下载未完成，仍缺少运行所需资源',
      retrying: '正在重试下载 {repository}',
      retryingMessage: '{repository} 下载失败，将在 {seconds} 秒后自动重试。',
      retryExceeded: '已自动重试 {limit} 次，最后一次错误：{error}',
      updatingRepository: '正在更新 {repository}',
      updatingRepositoryWithGit: '正在通过 Git 更新 {repository}。',
      repositoryUpdated: '{repository} 已更新',
      repositorySynced: '{repository} 已同步到最新版本。',
      installingRepository: '正在安装 {repository}',
      writingRepository: '正在写入 {repository} 到本地资源目录。',
      repositoryUpdatedToLatest: '{repository} 已更新到最新版本。',
      preparingRepositoryDownload: '正在准备下载 {repository}',
      preparingRepositoryDownloadMessage: '正在准备从 {url} 下载 {repository}。',
      downloadingRepository: '正在下载 {repository}',
      downloadingRepositoryFromGithub: '正在从 GitHub 下载 {repository} 资源包。',
      extractingRepository: '正在解压 {repository}',
      extractingRepositoryMessage: '正在解压 {repository} 资源包。',
      downloadingRepositoryWithGit: '正在通过 Git 下载 {repository} 资源包。',
    },
    repository: {
      assets: {
        'gakumasu-diff': 'gakumasu-diff',
        GakumasTranslationData: 'GakumasTranslationData',
      },
    },
  },
  backend: {
    api: {
      ok: 'OK',
      genericError: 'error',
      invalidTaskName: '任务不存在',
      taskConfigMissing: '该任务没有配置项。',
      shutdownStarted: '应用正在退出',
      resourceNotReady: '首次启动需要先下载游戏数据库和本地化资源，请在 WebUI 中确认下载。',
      gameDatabaseNotReady: '游戏数据库资源尚未就绪',
      taskQueueStartFailed: '任务队列启动失败',
      taskStartFailed: '任务启动失败',
      runFromFailed: '从当前任务开始执行失败',
      manualOnlyRunFromUnsupported: '仅手动任务不支持从这里开始执行',
      noRunningTask: '当前没有正在运行的任务',
      noSuspendedTask: '当前没有已挂起的任务',
      suspendUnsupported: '当前任务不支持手动挂起',
      resumeUnsupported: '当前任务不支持手动解除挂起',
      resumeBlockedByInsertedTask: '当前处于插队执行中，无法恢复执行',
      taskFailurePackageMissing: '日志压缩包不存在或已失效，请重试任务后重新下载。',
      refreshDmmTokenFailed: '提取游戏启动参数失败：{error}',
      imageDownloadDisabled: '游戏资源下载功能未启用，请在设置中开启',
      imageDownloadFeatureDisabledShort: '游戏资源下载功能未启用',
      objectManagerUnavailable: 'GkmasObjectManager 未就绪，请确认 vendor/GkmasObjectManager 子模块已初始化',
      objectManagerUnavailableShort: 'GkmasObjectManager 未就绪',
      downloadInProgress: '正在下载中，请稍后',
      supportCardThumbDownloadStarted: '开始下载支援卡缩略图',
      supportCardFullDownloadStarted: '开始下载支援卡全尺寸图片',
      downloadStarted: '开始下载',
      downloadAlreadyExists: '已存在',
      downloadAlreadyRunning: '下载已在进行中',
      supportCardAutoDownloadStarted: '开始自动下载支援卡图片',
      cardNotFound: '未找到卡牌：{cardId}',
    },
    app: {
      deviceInitializing: '正在初始化设备...',
      deviceUnavailable: '当前设备不可用。',
      deviceReadyAutoDetected: '已自动识别到可用设备',
      deviceDisconnected: '设备连接已断开',
      status: {
        ready: '设备已就绪',
        initializing: '正在初始化设备',
      },
    },
    device: {
      windows: {
        available: '',
        unavailable: {
          non_windows: 'PC 模式仅支持 Windows，请在 macOS/Linux 上使用 Phone 模式。',
          import_error: 'PC 模式依赖的 Windows 专用组件未就绪（通常是 pywin32 未安装或损坏），请重新执行 `pip install -r requirements.txt` 后重试。',
          unknown: 'PC 模式当前不可用。',
        },
      },
      mac: {
        available: '',
        unavailable: {
          non_macos: 'MacPlayTools 模式仅支持 macOS (Apple Silicon)。',
          import_error: 'MacPlayTools 模式依赖的组件未就绪：{error}',
          unknown: 'MacPlayTools 模式当前不可用。',
        },
      },
    },
    task: {
      startManual: '已开始手动执行任务：{task}',
      startFrom: '已从这里开始执行后续任务：{task}',
      enabled: '已启用任务：{task}',
      disabled: '已禁用任务：{task}',
      names: {
        start_game: '启动游戏',
        get_expenditure: '获取活动费',
        dispatch_work: '派遣任务',
        get_gift: '获取礼物/邮箱',
        auto_purchase: '自动每日交换',
        auto_enhancement_support_card: '自动强化支援卡',
        auto_contest: '自动每日竞技场',
        claim_task_rewards: '领取任务奖励',
        claim_pass_rewards: '领取通行证奖励',
        auto_producer: '自动培育（Beta）',
        void_task: '测试任务',
        refresh_skill_storage: '刷新技能卡存储',
        learn_support_card_clip: '刷新支援卡存储',
        learn_idol_card_clip: '刷新偶像卡存储',
      },
    },
    gameAsset: {
      downloading: '正在下载 {label}...',
      fetchingManifest: '正在获取资源清单...',
      fetchingManifestWithLabel: '正在获取资源清单（{label}）...',
      searchingWithLabel: '正在检索 {label} 资源...',
      noMatchingResources: '未找到 {label} 资源',
      noMatchingObjects: 'no matching objects',
      downloadCompleted: '{label}下载完成：{downloaded} 张新图片，{skipped} 张已跳过（共缓存 {total} 张）',
      phaseCompleted: '{label}：{downloaded} 张新图片，{skipped} 张已跳过（共缓存 {total} 张）',
      downloadFailed: '{label}下载失败',
      downloadFailedError: '{error}',
      dialogAssetsCompleted: '支援卡相关资源全部下载完成',
      bulkDownloadFailed: '资源下载失败',
    },
    message: {
      gameNotForegroundRetry: '游戏未在前台运行，请手动切回游戏后重试',
      gameNotStartedRetry: '游戏未启动，请先手动启动游戏后重试',
      gameNotForegroundStart: '游戏未在前台运行，请手动启动游戏后重试',
      startLearningIdolClip: '开始偶像卡 CLIP 学习，将按底部卡条逐张遍历',
    },
    config: {
      section: {
        base: '基础设置',
        dmm_player: 'DMM Player',
      },
      base: {
        run_mode: {
          label: '运行模式',
          hint: '脚本的执行模式（需重启生效）',
          option: {
            pc: '电脑端（DMM）',
            phone: '手机端',
            mac_play_tools: 'macOS PlayCover',
          },
          disabledReason: {
            pcWindowsOnly: 'PC / DMM 模式仅支持 Windows。',
            macOnly: 'MacPlayTools 模式仅支持 macOS (Apple Silicon)。',
          },
        },
        ocr_backend: {
          label: 'OCR 后端',
          hint: 'auto：macOS 优先 Vision，其他平台使用 RapidOCR；失败时会回退到 RapidOCR（修改后需重启生效）',
          option: {
            auto: '自动',
            rapidocr: 'RapidOCR',
            vision: 'Vision（macOS 原生 OCR）',
          },
          disabledReason: {
            visionMacOnly: 'Vision OCR 仅在 macOS 可用。',
          },
        },
        adb_connect_mode: {
          label: 'ADB连接模式',
          hint: '安卓调试桥的连接模式，手机建议使用USB，模拟器可使用网络连接（修改后需重启生效）',
          option: {
            network: {
              title: '网络连接',
            },
            u_s_b: {
              title: 'USB连接',
            },
          },
        },
        adb_host: {
          label: 'ADB主机名',
          hint: '安卓调试桥的ip地址，模拟器一般是127.0.0.1',
        },
        adb_port: {
          label: 'ADB端口',
          hint: '安卓调试桥的端口，默认5555，Android11以上为系统随机',
        },
        adb_serial: {
          label: '通过USB连接的ADB设备',
          hint: '请选择通过USB连接的设备，如未找到设备请尝试刷新列表',
        },
        android_screen_capture_service: {
          label: 'ADB截图方式',
          hint: 'scrcpy / DroidCast 的延迟通常优于 ADB 截图；scrcpy 需要将官方 Releases 的 scrcpy-server 放到 bin',
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
          label: 'ADB点击屏幕方式',
          hint: '可选 MaaTouch / minitouch / scrcpy；MaaTouch 需放入官方构建产物到 bin/maatouch 或用 workflow 生成，minitouch 需放入官方构建产物到 bin/minitouch 且当前只支持 Android 9 及以下',
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
          label: '自动启动游戏',
          hint: '当游戏未启动时是否自动启动游戏',
        },
        auto_startup_time: {
          label: '自动运行触发时间',
          hint: '24小时制，格式为 HH:MM',
        },
        battle_decision_backend: {
          label: '战斗决策后端',
          hint: 'レッスン/試験（Lesson/Exam）阶段的出牌决策方式',
          option: {
            algo: {
              title: '简单算法决策',
            },
            llm: {
              title: 'LLM（Bata）',
            },
            rl_battle: {
              title: 'RL（WIP）',
            },
          },
        },
        check_resource_updates_on_startup: {
          label: '启动时检查资源仓库更新',
          hint: '每次启动后立即检查一次资源仓库更新',
        },
        disabled_tasks: {
          label: '禁用任务列表',
          hint: '配置禁用任务列表',
        },
        enable_game_asset_download: {
          label: '是否启用游戏资源下载',
          hint: '使用GkmasObjectManager从游戏服务器下载游戏资源文件。需要有互联网连接。',
        },
        enabled_auto_startup: {
          label: '每日自动执行脚本',
          hint: '开启后会在设定时间自动开始执行任务队列',
        },
        enabled_check_resource_updates: {
          label: '定时检查资源仓库更新',
          hint: '按设定周期检查 assets/GakumasTranslationData 与 assets/gakumasu-diff 是否有上游更新',
        },
        gakumas_translation_data_repository_url: {
          label: 'GakumasTranslationData 仓库 URL',
          hint: '资源下载与更新使用的仓库地址。修改后会立即重新检查更新状态。（如果你不知道这是什么，请不要更改）',
        },
        gakumasu_diff_repository_url: {
          label: 'gakumasu-diff 仓库 URL',
          hint: '资源下载与更新使用的仓库地址。修改后会立即重新检查更新状态。（如果你不知道这是什么，请不要更改）',
        },
        game_package_name: {
          label: '游戏包名',
          hint: '默认：com.bandainamcoent.idolmaster_gakuen（修改后需重启生效）',
        },
        game_window_name: {
          label: '游戏窗口名',
          hint: '默认：gakumas（修改后需重启生效）',
        },
        llm_api_key: {
          label: 'LLM API Key',
          hint: 'API 密钥',
        },
        llm_base_url: {
          label: 'LLM API 地址',
          hint: 'OpenAI 兼容 API 端点（llama / vLLM / OpenAI 等）',
        },
        llm_insight_api_key: {
          label: '洞察模型 API Key',
          hint: '为空则使用主 LLM Key',
        },
        llm_insight_base_url: {
          label: '洞察模型 API 地址',
          hint: '为空则使用主 LLM 地址；可配置独立模型（云端/小模型）',
        },
        llm_insight_enabled: {
          label: '启用策略洞察',
          hint: '后台生成可迁移的策略洞察，供后续决策参考',
        },
        llm_insight_max_tokens: {
          label: '洞察模型最大输出 Token',
          hint: '0=不限制，让模型自行管理 thinking + output 的 token 分配',
        },
        llm_insight_model: {
          label: '洞察模型',
          hint: '为空则使用主 LLM 模型名称',
        },
        llm_insight_num_ctx: {
          label: '洞察模型上下文窗口',
          hint: '0=不设置，由 API 自动管理',
        },
        llm_insight_reasoning_effort: {
          label: '洞察模型思考强度',
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
          label: '洞察模型温度',
          hint: '洞察生成的温度，越低越确定（0.0 ~ 1.0）',
        },
        llm_insight_timeout: {
          label: '洞察模型超时(秒)',
          hint: '后台洞察生成的超时时间，可比主决策更长',
        },
        llm_max_tokens: {
          label: 'LLM 最大输出 Token',
          hint: '输出 token 上限（包含思考+回答），设为 0 表示自动（不传给 API）',
        },
        llm_model: {
          label: 'LLM 模型',
          hint: '模型名称（如 gpt-oss:20b、qwen3:4b、qwen3.5:9b 等）',
        },
        llm_num_ctx: {
          label: 'LLM 上下文窗口',
          hint: '可选兼容参数，主要供 Ollama / 本地 OpenAI 兼容后端覆盖上下文窗口；设为 0 表示自动',
        },
        llm_reasoning_effort: {
          label: 'LLM 思考强度',
          hint: '控制推理深度；low=更快，medium=平衡，high=更充分',
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
          label: 'LLM 温度',
          hint: '生成温度，越低越确定（0.0 ~ 1.0）',
        },
        llm_timeout: {
          label: 'LLM 超时(秒)',
          hint: 'API 请求超时时间',
        },
        other_decision_backend: {
          label: '其他决策后端',
          hint: '对话/P饮料/技能奖励/咨询/道具选择等阶段的决策方式',
          option: {
            algo: {
              title: '简单算法决策',
            },
            llm: {
              title: 'LLM（Bata）',
            },
            rl_battle: {
              title: 'RL（WIP）',
            },
          },
        },
        playtools_port: {
          label: 'PlayTools 端口',
          hint: 'PlayCover 游戏窗口标题栏 [localhost:端口号] 中的端口号（修改后需重启生效）',
        },
        prefer_game_asset_image: {
          label: '在 UI 中始终使用游戏资源显示',
          hint: '启用后，UI 中的物品/支援卡/技能卡图片将始终优先使用从游戏服务器下载的资源图片，而非游戏过程中截图识别的图像。',
        },
        resource_update_check_period: {
          label: '资源仓库检查周期',
          hint: '仅用于定时检查',
          option: {
            daily: {
              title: '每天',
            },
            every_3_days: {
              title: '每 3 天',
            },
            weekly: {
              title: '每周',
            },
          },
        },
        rl_inference_base_url: {
          label: 'RL 推理服务地址',
          hint: '无状态 RL 推理服务地址',
        },
        rl_inference_timeout: {
          label: 'RL 推理超时(秒)',
          hint: '请求 RL 服务的超时时间',
        },
        schedule_decision_backend: {
          label: '周行程决策后端',
          hint: '周行程（Schedule）阶段的自动决策方式',
          option: {
            algo: {
              title: '简单算法决策',
            },
            llm: {
              title: 'LLM（Bata）',
            },
            rl_battle: {
              title: 'RL（WIP）',
            },
          },
        },
      },
      dmm_player: {
        game_exe_path: {
          label: '游戏安装目录',
          hint: '游戏安装路径，指向gakumas.exe（默认自动获取，非必要无需修改）',
        },
        viewer_id: {
          label: 'Viewer ID',
          hint: '自动获取，非必要无需修改',
        },
        open_id: {
          label: 'Open ID',
          hint: '自动获取，非必要无需修改',
        },
        pf_token: {
          label: 'PF Token',
          hint: '自动获取，非必要无需修改',
        },
      },
      task__auto_purchase: {
        weekly_gift: {
          label: '购买每周礼包',
          hint: '每日检查礼包页面是否有免费可购买项',
        },
        daily_buy_list: {
          label: '每日购买物品',
          hint: '从交换所中选择允许自动购买的物品',
        },
        refresh_shop: {
          label: '自动刷新交换所',
          hint: '每日自动刷新交换所',
        },
        use_gem_refresh: {
          label: '使用钻石刷新交换所',
          hint: '免费刷新后仍可用钻石继续刷新',
        },
      },
      task__auto_contest: {
        auto_reconfigure_team_before_challenge: {
          label: '挑战前自动重新配置队伍',
          hint: '如果队伍中有空位仍会触发自动配置',
        },
        challenge_order: {
          label: '挑战顺序',
          hint: '脚本会按设定顺序寻找符合条件的挑战对象',
          option: {
            random: {
              title: '随机选择',
            },
            highest_power: {
              title: '最高',
            },
            lowest_power: {
              title: '最低',
            },
            balanced_power: {
              title: '中间',
            },
          },
        },
      },
      task__dispatch_work: {
        reconfigure_work_hours: {
          label: '重新配置任务派遣时间',
          hint: '开启后会在派遣前重新设置工作时长',
        },
        working_hours: {
          label: '任务派遣时间',
          hint: '仅在开启“重新配置任务派遣时间”时生效',
          option: {
            '4_h': {
              title: '4小时（最低）',
            },
            '8_h': {
              title: '8小时',
            },
            '12_h': {
              title: '12小时（最高）',
            },
          },
        },
      },
      task__auto_enhancement_support_card: {
        enhance_r: {
          label: '强化 R 卡',
          hint: '自动强化 R 品级的支援卡',
        },
        enhance_r_max_level: {
          label: 'R 最大强化等级',
          hint: 'R 卡的最大目标等级',
        },
        enhance_sr: {
          label: '强化 SR 卡',
          hint: '自动强化 SR 品级的支援卡',
        },
        enhance_sr_max_level: {
          label: 'SR 最大强化等级',
          hint: 'SR 卡的最大目标等级',
        },
        enhance_ssr: {
          label: '强化 SSR 卡',
          hint: '自动强化 SSR 品级的支援卡',
        },
        enhance_ssr_max_level: {
          label: 'SSR 最大强化等级',
          hint: 'SSR 卡的最大目标等级',
        },
        auto_limit_break: {
          label: '自动执行上限解放',
          hint: '有同名卡片且未达到星级上限时，自动进行上限解放',
        },
        auto_convert: {
          label: '自动交换溢出的支援卡',
          hint: '自动将溢出的支援卡变换为「サポートの証」',
        },
        whitelist_mode: {
          label: '白名单模式',
          hint: '仅强化白名单中选择的卡牌',
        },
        whitelist_card_ids: {
          label: '白名单卡片',
          hint: '选择允许被自动强化的支援卡',
        },
        whitelist: {
          no_selection: '尚未选择白名单支援卡',
          open_dialog: '选择白名单卡片',
          dialog_title: '支援卡白名单',
          search_placeholder: '搜索支援卡名称（支持中文/日文/ID）',
          downloading_images: '正在下载支援卡相关图片...',
          add_to_whitelist: '加入白名单',
          levelShort: 'Lv',
          limit_only: '限定',
          trigger_rate: '支援发生率',
          filter: {
            rarity: '稀有度',
            type: '类型',
            plan: '路线',
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
            stamina_short: '体',
            assist_short: '辅',
          },
          plan: {
            plan1: '感性',
            plan2: '逻辑',
            plan3: '异常',
            common: '通用',
          },
          card_category: {
            active_skill: '主动技能',
            mental_skill: '精神技能',
            trouble: '麻烦卡',
            free_skill: '自由技能',
            skill_card: '技能卡',
            p_item: 'P物品',
          },
          section: {
            support_ability: '支援能力',
            support_event: '支援事件',
            attachments: '附带奖励',
          },
        },
      },
      task__auto_producer: {
        scenario: {
          label: '剧本',
          hint: '选择培育剧本',
          option: {
            hajime: {
              title: '初',
            },
            nia: {
              title: 'NIA（WIP）',
            },
          },
        },
        difficulty: {
          label: '难度',
          hint: '选择培育难度',
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
          label: 'NIA 难度',
          hint: '选择 NIA 剧本难度',
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
          label: '目标偶像卡',
          hint: '目标 P アイドル ID（留空使用默认选中的卡；需先执行「刷新偶像卡存储」学习卡片特征）',
        },
        support_card_mode: {
          label: '支援卡编成',
          hint: '自动编成或使用预设编号',
          option: {
            auto: {
              title: '自动编成',
            },
            preset: {
              title: '预设编号',
            },
          },
        },
        support_card_preset_index: {
          label: '支援卡预设编号',
          hint: '使用第几组预设编成',
        },
        memory_mode: {
          label: '记忆编成',
          hint: '自动编成或使用预设编号',
          option: {
            auto: {
              title: '自动编成',
            },
            preset: {
              title: '预设编号',
            },
          },
        },
        memory_preset_index: {
          label: '记忆预设编号',
          hint: '使用第几组预设编成',
        },
        use_rental: {
          label: '使用租赁记忆',
          hint: '自动编排记忆时勾选「レンタルを使用」复选框',
        },
        use_boost_items: {
          label: '使用加成道具',
          hint: '「開始確認」页面是否使用加成道具（編成詳细按钮上方）',
        },
        resume_interrupted: {
          label: '恢复中断培育',
          hint: '检测到上次中断的培育时自动恢复（点击「再開する」），而非放弃重新开始',
        },
        allow_ap_recovery: {
          label: '是否允许使用道具恢复AP',
          hint: 'AP 不足时是否允许自动消耗道具恢复',
        },
        allow_destroy_production_data: {
          label: '是否允许销毁跨设备未完成培育的会话',
          hint: '检测到「プロデュースデータの破棄」时是否允许确认继续',
        },
        schedule_notebook_mode: {
          label: 'P手帐读取策略',
          hint: 'disabled：不读取；before_decision：仅在周行动自动决策前读取（决策后不再触发）',
          option: {
            disabled: {
              title: '关闭读取',
            },
            before_decision: {
              title: '仅决策前读取',
            },
          },
        },
        memory_photo_mode: {
          label: '记忆卡面选择',
          hint: '培育结束后选择记忆卡面的方式',
          option: {
            first: {
              title: '默认选择第一个',
            },
            vl: {
              title: 'VL 视觉模型自动选择最优卡面',
            },
          },
        },
        memory_photo_vl_prompt: {
          label: 'VL 选卡面提示词',
          hint: '自定义 VL 模型选择卡面时使用的提示词（留空使用默认）',
        },
        idol_card_browser: {
          no_selection: '尚未选择目标偶像卡',
          search_placeholder: '搜索偶像卡名称（支持中文/日文/ID）',
          limit_only: '限定',
          total: '总计',
          stamina: '体力',
          after_training: '培育后',
          stat: {
            vocalShort: 'Vo',
            danceShort: 'Da',
            visualShort: 'Vi',
          },
          rarity: {
            ssr: 'SSR',
            sr: 'SR',
            r: 'R',
          },
          plan: {
            plan1: '感性',
            plan2: '逻辑',
            plan3: '异常',
            common: '通用',
          },
          attribute: {
            vocal: '声乐',
            dance: '舞蹈',
            visual: '表现',
          },
          exam_effect: {
            parameter_buff: '参数提升',
            review: '复习',
            lesson_buff: '课程加成',
            concentration: '专注',
            card_play_aggressive: '攻击性出牌',
            full_power: '全力',
          },
          filter: {
            rarity: '稀有度',
            plan: '路线',
            attribute: '属性',
            exam_effect: '考试效果',
            character: '角色',
          },
          section: {
            growth: '成长属性',
            skill_card: '技能卡',
            item: '道具',
          },
          skin: {
            before: '觉醒前',
            after: '觉醒后',
          },
        },
      },
    },
    adb: {
      missing: '未安装 adb，请先安装 Android SDK Platform-Tools 并将 adb 加入 PATH。',
      noError: '',
      notConnected: '当前未连接到 ADB 设备。',
      invalidConnectMode: '无效的 ADB 连接模式：{mode}',
      deviceDisconnectedTarget: 'ADB 设备 {target} 已断开或未连接。请确认模拟器/设备正在运行。',
      deviceDisconnectedSerial: 'ADB 设备 {serial} 已断开或未连接。请确认 USB 已连接、已开启 USB 调试，并在 WebUI 中刷新设备列表。',
      deviceDisconnected: 'ADB 设备已断开或未连接。请确认设备在线后重试。',
      deviceOfflineTarget: 'ADB 设备 {target} 当前处于离线状态。请重试或重启设备。',
      deviceOfflineSerial: 'ADB 设备 {serial} 当前处于离线状态。请重新插拔设备并确认 USB 调试授权。',
      deviceOffline: 'ADB 设备当前处于离线状态。请重新连接设备后重试。',
      usbNotFoundSerial: '未找到所选 USB ADB 设备：{serial}。请确认设备已连接、已开启 USB 调试，并在 WebUI 中刷新设备列表后重新选择。',
      usbNotFound: '未检测到可用的 USB ADB 设备。请连接设备、开启 USB 调试，并在 WebUI 中刷新设备列表。',
      networkUnavailable: '未能连接到 ADB 设备 {target}。请确认 adb 已安装、设备已联网并已执行 adb tcpip / adb connect，原始错误：{message}',
      initFailed: 'ADB 初始化失败：{message}',
    },
  },
  config: {
    loadingHint: '如长时间未显示请重启后端服务以应用配置更新。',
    unavailableOptionsPrefix: '当前不可选：{items}',
    optionDisabledItem: '{title}：{reason}',
    componentNotRegistered: '未注册的配置组件：{component}',
  },
  logger: {
    close: '关闭日志',
    open: '打开日志',
    title: '执行日志',
  },
  api: {
    errorPrefix: 'API错误：{message}{status}',
    statusSuffix: ' (status:{status})',
  },
}

export default zhHans
