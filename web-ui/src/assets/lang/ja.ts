import { ja as vuetifyJa } from 'vuetify/locale'
import zhHans from './zhHans'
import { deepMergeMessages } from '@/scripts/i18n/messageTools'

/**
 * 日本語言語パック。
 */
const ja = deepMergeMessages(zhHans, {
  $vuetify: vuetifyJa,
  app: {
    sections: {
      tasks: 'タスク一覧',
      settings: '設定',
      about: 'このプロジェクトについて',
    },
    footer: {
      exit: 'アプリを終了',
      exiting: 'アプリを終了しています...',
      teamName: 'Pigeon Server Team',
      licenseLabel: 'GPLv3 ライセンス',
      closeLogger: 'ログを閉じる',
      openLogger: 'ログを開く',
      executionLog: '実行ログ',
    },
    window: {
      minimize: '縮小',
      maximize: '拡大',
      restore: '元に戻す',
      close: '閉じる',
    },
    preferences: {
      languageMenu: '言語を切り替え',
      themeMenu: 'テーマを切り替え',
    },
  },
  common: {
    confirm: '確認',
    cancel: 'キャンセル',
    save: '設定を保存',
    reset: '初期設定に戻す',
    loading: '読み込み中',
    notRun: '未実行',
    yes: 'はい',
    no: 'いいえ',
    unknown: '不明',
    close: '閉じる',
    refresh: '更新',
    retry: '再試行',
    auto: '自動',
    currentTask: '現在のタスク',
    manualOnly: '手動のみ',
    system: 'システム設定',
    light: 'ライト',
    dark: 'ダーク',
  },
  dialogs: {
    resetSettingsTitle: 'すべての設定をリセットしますか？',
    resetSettingsText: 'この操作を行うと、タスク設定を含むすべての設定が初期値に戻ります。',
    runFromTitle: 'ここから実行しますか？',
    runFromDescription: '「{task}」から開始し、一覧順に有効な自動タスクを続けて実行します。',
    taskError: {
      title: 'タスクの実行に失敗しました',
      unknownTask: '不明なタスク',
      description: 'タスク {task} の実行に失敗しました。GitHub Issues または QQ グループにログを送っていただくと、原因の特定がしやすくなります。',
      exception: '例外: {errorType} {errorMessage}',
      downloadLog: 'ログ圧縮ファイルをダウンロード',
      copyQqGroup: 'QQ グループ番号をコピー',
      openGithub: 'GitHub フィードバックを開く',
      acknowledge: '了解',
    },
    selector: {
      chooseCard: 'カードを選択',
      selectCard: 'カードを選択',
      cancelSelect: '選択を取り消す',
      searchPlaceholder: 'カード名を検索（中国語/日本語/ID 対応）',
      selectedCount: '{count} 件選択中',
      totalCards: '{count} 枚',
      noMatchedCards: '一致するカードがありません',
      clearSelection: '選択をクリア',
      confirmSingle: '確認',
      confirmMulti: '確認 ({count})',
    },
  },
  websocket: {
    disconnected: '接続が切断されました',
    reconnected: 'サーバーへの再接続に成功しました',
    waitingServer: 'サーバーの応答を待っています.....',
    invalidBinaryFormat: '無効なバイナリ形式です',
  },
  toolbar: {
    deviceNotReady: 'デバイスの準備ができていません',
    deviceRetryHint: '{message} 「開始」を押すと、デバイスへの再接続をもう一度試みます。',
    waitingAction: '待機中',
    running: 'スクリプト実行中......',
    suspended: 'スクリプト停止中......',
    start: '開始',
    stop: '停止',
    suspend: '一時停止',
    resume: '再開',
    startQueued: 'タスクキューを開始しました',
    stopQueued: 'タスクキューを停止しています',
    suspendedDone: 'タスクを一時停止しました',
    resumedDone: 'タスクを再開しました',
  },
  settings: {
    title: 'スクリプト設定',
    basicSection: '基本設定',
    language: {
      label: '表示言語',
      hint: 'WebUI の表示言語を切り替えます',
      option: {
        system: 'システムに合わせる',
        zhHans: '简体中文',
        zhHant: '繁體中文',
        en: 'English',
        ja: '日本語',
      },
    },
    theme: {
      label: '表示テーマ',
      option: {
        system: 'システムに合わせる',
        light: 'ライトモード',
        dark: 'ダークモード',
      },
    },
    saveSuccess: '設定を保存しました',
    resetSuccess: '設定をリセットしました。一部の設定は再起動後に反映されます。',
    refreshLaunchArgs: '起動パラメータを更新',
    refreshLaunchArgsSuccess: '起動パラメータを更新しました',
    resourceUpdate: {
      title: 'リソース更新',
      bootstrapAction: '必要なリソースをダウンロード',
      applyAction: '今すぐ更新',
      checkActionIdle: '更新を確認',
      checkActionBusy: '確認中',
      state: {
        pending: '未確認',
        checked: '確認済み',
        checking: '確認中',
        downloading: 'ダウンロード中',
        updating: '更新中',
        updateAvailable: '更新あり',
        bootstrapPending: '未ダウンロード',
        error: '確認失敗',
      },
      headline: {
        bootstrapIdle: 'このパッケージにはゲームデータベースとローカライズ資産が同梱されなくなったため、初回起動時にダウンロードが必要です。',
        bootstrapRunning: '初回起動に必要なゲームデータベースとローカライズ資産をダウンロードしています',
        updating: 'リソースリポジトリを同期し、ゲームデータベースを再読み込みしています',
        checking: 'GakumasTranslationData と gakumasu-diff の上流更新を確認しています',
        hasUpdate: '新しいリソースリポジトリのバージョンがあります',
        lastError: '直近の確認でエラーが発生しました',
        checked: '現在のリソースリポジトリ状態は同期済みです',
        idle: '手動確認するか、起動時/定期確認を待てます',
      },
      bootstrapNotice: '初回起動前にゲームデータベースとローカライズ資産のダウンロードが必要です。失敗した場合は自動で再試行し、完了後に初期化を続行します。',
      progressFallbackTitle: 'リソース処理中',
      progress: {
        remoteHeadNotFound: 'リモート HEAD を取得できません',
        gitNotFound: 'Git 実行可能ファイルが見つかりません',
      },
      progressFallbackMessage: 'リソースを同期しています。しばらくお待ちください。',
      recentError: '直近のエラー: {error}',
      pendingDialogTitle: '初回起動前にリソースのダウンロードが必要です',
      pendingDialogDescription: 'このパッケージにはゲームデータベースとローカライズ資産が同梱されなくなりました。確認後、自動的にダウンロードを開始し、その後初期化を続行します。',
      pendingDialogMeta: '初回起動前にゲームデータベースとローカライズ資産のダウンロードが必要です。',
      pendingDialogMetaItem: '{name}（不足 {missing}/{required} ファイル）',
      dialogAgree: '同意してダウンロード開始',
      dialogLater: '後で',
      downloadProgressTitle: 'リソースをダウンロードしています',
    },
  },
  tasks: {
    title: 'タスク一覧',
    blockedHint: 'リソースのダウンロード完了前でもタスクと設定は確認できますが、実行はできません。',
    taskName: 'タスクID: ',
    enabled: '有効: ',
    lastRunTime: '前回実行: ',
    run: '実行',
    runFrom: 'ここから実行',
    disable: '無効化',
    enable: '有効化',
    settings: 'タスク設定',
    runBlockedTitle: 'リソースの準備ができていないため実行できません',
    runBusyTitle: 'すでに実行中のタスクキューがあります',
    runFromTitle: 'このタスクから後続の有効な自動タスクを実行します',
    status: {
      PENDING: '待機中',
      RUNNING: '実行中',
      SUSPENDED: '停止中',
      SUCCESS: '完了',
      FAILED: '失敗',
      CANCELED: 'キャンセル済み',
      UNKNOWN: '不明',
    },
    relativeTime: {
      justNow: 'たった今',
      secondsAgo: '{count}秒前',
      minutesAgo: '{count}分前',
      hoursAgo: '{count}時間前',
      yesterday: '昨日',
      daysAgo: '{count}日前',
    },
  },
  resource: {
    progress: {
      step: 'ステップ {current}/{total}',
      attempt: '試行 {current}/{total}',
      retryInSeconds: '{seconds}秒後に再試行',
      lastCheckedAt: '最終確認: {time}',
      nextCheckAt: '次回定期確認: {time}',
      bootstrapDownloadHint: '初回起動前にゲームデータベースとローカライズ資産のダウンロードが必要です',
      checkingHint: 'リソースリポジトリの更新を確認しています',
      updatingHint: 'リソースリポジトリを更新しています',
      notCheckedYet: 'まだリソースリポジトリの確認は実行されていません',
      checkFailed: 'リソース確認に失敗しました: {error}',
      checkHasUpdate: 'リソースリポジトリに更新があります',
      checkUpToDate: 'リソースリポジトリは最新です',
      updateFailed: 'リソース更新に失敗しました: {error}',
      bootstrapCompleted: '初回起動に必要なリソースのダウンロードが完了し、ゲームデータベースを再読み込みしました。',
      updateCompleted: 'リソースリポジトリの更新が完了し、ゲームデータベースを再読み込みしました。',
      bootstrapPromptSingle: '初回起動前にゲームデータベースとローカライズ資産のダウンロードが必要です。進捗は表示され、失敗時は自動再試行します。今すぐ開始しますか？',
      bootstrapPromptMultiple: '初回起動前に次のリソースが必要です: {repositories}。進捗は表示され、失敗時は自動再試行します。今すぐ開始しますか？',
      updatePromptSingle: 'リソースリポジトリに更新があります。今すぐ更新してゲームデータベースを再読み込みしますか？',
      updatePromptMultiple: '次のリソースリポジトリに更新があります: {repositories}。今すぐ更新してゲームデータベースを再読み込みしますか？',
      repoMissingItem: '{name}（不足 {missing}/{required} ファイル）',
      repositoryCommitRange: '{name}（{local} -> {remote}）',
      updatePromptTitle: 'リソースリポジトリの更新を検出しました',
      updatePromptConfirm: '今すぐ更新',
      updatePromptCancel: '後で',
      checkCompleted: 'リソースリポジトリの確認が完了しました',
      bootstrapRunning: '初回起動に必要なリソースをダウンロードしています...',
      updateRunning: 'リソースリポジトリを更新しています...',
      checkCompletedWithErrors: 'リソースリポジトリの確認は完了しましたが、一部のリポジトリの確認に失敗しました: {error}',
      operationLocked: '現在リソースリポジトリの確認または更新を実行中です',
      taskRunning: 'タスク実行中のため、リソースリポジトリを更新できません',
      upToDate: 'リソースリポジトリはすでに最新です',
      reloadingDatabase: 'ゲームデータベースを再読み込みしています',
      reloadingDatabaseMessage: 'リソースのダウンロードが完了しました。ゲームデータベースと関連サービスを再読み込みしています。',
      reloadFailed: 'リソースの再読み込みに失敗しました: {error}',
      repositoryError: '{error}',
      noError: '',
      resourcesMissing: 'リソースのダウンロードが未完了で、実行に必要なリソースがまだ不足しています。',
      retrying: '{repository} のダウンロードを再試行しています',
      retryingMessage: '{repository} のダウンロードに失敗しました。{seconds} 秒後に自動で再試行します。',
      retryExceeded: '{limit} 回自動再試行しました。最後のエラー: {error}',
      updatingRepository: '{repository} を更新しています',
      updatingRepositoryWithGit: 'Git で {repository} を更新しています。',
      repositoryUpdated: '{repository} を更新しました',
      repositorySynced: '{repository} は最新バージョンに同期されました。',
      installingRepository: '{repository} をインストールしています',
      writingRepository: '{repository} をローカルのリソースディレクトリに書き込んでいます。',
      repositoryUpdatedToLatest: '{repository} を最新バージョンに更新しました。',
      preparingRepositoryDownload: '{repository} のダウンロードを準備しています',
      preparingRepositoryDownloadMessage: '{url} から {repository} をダウンロードする準備をしています。',
      downloadingRepository: '{repository} をダウンロードしています',
      downloadingRepositoryFromGithub: 'GitHub から {repository} のアーカイブをダウンロードしています。',
      extractingRepository: '{repository} を展開しています',
      extractingRepositoryMessage: '{repository} のアーカイブを展開しています。',
      downloadingRepositoryWithGit: 'Git で {repository} のアーカイブをダウンロードしています。',
    },
  },
  backend: {
    api: {
      ok: 'OK',
      genericError: 'error',
      invalidTaskName: 'タスクが存在しません',
      taskConfigMissing: 'このタスクには設定項目がありません。',
      shutdownStarted: 'アプリを終了しています',
      resourceNotReady: '初回起動前にゲームデータベースとローカライズ資産のダウンロードが必要です。WebUI で確認してください。',
      gameDatabaseNotReady: 'ゲームデータベースのリソースが未準備です',
      taskQueueStartFailed: 'タスクキューの開始に失敗しました',
      taskStartFailed: 'タスクの開始に失敗しました',
      runFromFailed: 'このタスクからの開始に失敗しました',
      manualOnlyRunFromUnsupported: '手動専用タスクはキュー開始位置にできません',
      noRunningTask: '現在実行中のタスクはありません',
      noSuspendedTask: '現在停止中のタスクはありません',
      suspendUnsupported: '現在のタスクは手動停止に対応していません',
      resumeUnsupported: '現在のタスクは手動再開に対応していません',
      resumeBlockedByInsertedTask: '割り込みタスク実行中のため、いまは再開できません',
      taskFailurePackageMissing: 'ログ圧縮ファイルが存在しないか期限切れです。タスクを再実行してから再度ダウンロードしてください。',
      refreshDmmTokenFailed: 'ゲーム起動パラメータの取得に失敗しました: {error}',
      imageDownloadDisabled: 'ゲーム資産ダウンロード機能が無効です。設定で有効にしてください。',
      imageDownloadFeatureDisabledShort: 'ゲーム資産ダウンロード機能が無効です',
      objectManagerUnavailable: 'GkmasObjectManager の準備ができていません。vendor/GkmasObjectManager サブモジュールが初期化されているか確認してください。',
      objectManagerUnavailableShort: 'GkmasObjectManager の準備ができていません',
      downloadInProgress: 'ダウンロード中です。しばらくお待ちください。',
      supportCardThumbDownloadStarted: 'サポートカードのサムネイル画像のダウンロードを開始しました',
      supportCardFullDownloadStarted: 'サポートカードのフルサイズ画像のダウンロードを開始しました',
      downloadStarted: 'ダウンロードを開始しました',
      downloadAlreadyExists: 'すでに存在します',
      downloadAlreadyRunning: 'ダウンロードはすでに実行中です',
      supportCardAutoDownloadStarted: 'サポートカード画像の自動ダウンロードを開始しました',
      cardNotFound: 'カードが見つかりません: {cardId}',
    },
    app: {
      deviceInitializing: 'デバイスを初期化しています...',
      deviceUnavailable: '現在のデバイスは利用できません。',
      deviceReadyAutoDetected: '利用可能なデバイスを自動検出しました',
      deviceDisconnected: 'デバイス接続が切断されました',
      status: {
        ready: 'デバイス準備完了',
        initializing: 'デバイス初期化中',
      },
    },
    device: {
      windows: {
        available: '',
        unavailable: {
          non_windows: 'PC モードは Windows のみ対応しています。macOS / Linux では Phone モードを使用してください。',
          import_error: 'PC モードに必要な Windows 専用コンポーネントの準備ができていません（通常は pywin32 が未インストールまたは破損しています）。`pip install -r requirements.txt` を再実行してから再試行してください。',
          unknown: 'PC モードは現在利用できません。',
          init_failed: 'Windows デバイスの初期化に失敗しました: {message}',
        },
      },
      mac: {
        available: '',
        unavailable: {
          non_macos: 'MacPlayTools モードは macOS（Apple Silicon）のみ対応しています。',
          import_error: 'MacPlayTools モードに必要なコンポーネントの準備ができていません: {error}',
          unknown: 'MacPlayTools モードは現在利用できません。',
          port_not_configured: 'MacPlayTools ポートが未設定です。PlayCover でゲームを起動後、ウィンドウタイトルバーの [localhost:ポート番号] からポート番号を取得し、設定に入力してください。',
          connect_failed: 'MacPlayTools ({host}:{port}) に接続できません。PlayCover でゲームが起動しており、MaaTools が有効になっていることを確認してください。',
          init_failed: 'MacPlayTools デバイスの初期化に失敗しました: {message}',
        },
      },
    },
    task: {
      startManual: '手動タスクを開始しました: {task}',
      startFrom: 'ここから後続タスクを開始しました: {task}',
      enabled: 'タスクを有効化しました: {task}',
      disabled: 'タスクを無効化しました: {task}',
      names: {
        start_game: 'ゲーム起動',
        get_expenditure: '活動費を回収',
        dispatch_work: 'お仕事派遣',
        get_gift: 'ギフト/メール回収',
        auto_purchase: 'デイリー交換所自動化',
        auto_enhancement_support_card: 'サポートカード自動強化',
        auto_contest: 'デイリーコンテスト自動化',
        claim_task_rewards: 'タスク報酬受取',
        claim_pass_rewards: 'パス報酬受取',
        auto_producer: '自動プロデュース（Beta）',
        void_task: 'テストタスク',
        refresh_skill_storage: 'スキルカード保存を更新',
        learn_support_card_clip: 'サポートカード保存を更新',
        learn_idol_card_clip: 'アイドルカード保存を更新',
      },
      gameNotInForeground: 'ゲームがフォアグラウンドにありません。手動でゲームに切り替えて再試行してください。',
      gameNotStarted: 'ゲームが起動していません。先にゲームを手動で起動して再試行してください。',
      suspendSwitchToAlbum: 'タスクが一時停止しました。手動で図鑑ページに切り替えてください。',
      tabBarNotFound: 'TabBarが見つかりません。更新に失敗しました。',
      requiredButtonNotFound: '必要なボタンが見つかりません。更新に失敗しました。',
      cardAttributeSwitchNotFound: 'カード属性切替ボタンが見つかりません。更新に失敗しました。',
      idolSwitchNotFound: 'アイドル切替ボタンが見つかりません。更新に失敗しました。',
      startIdolCardCLIP: 'アイドルカード CLIP 学習を開始します。下部のカードバーに沿って順次遍歴します。',
      startSupportCardCLIP: 'サポートカード CLIP 学習を開始します。サポートカード一覧ページにいることを確認してください。',
      noEnhancementConfigured: '強化対象の品級が設定されていません。',
      clipNotInitialized: 'CLIP サービスが初期化されていません。アイドルカード CLIP を学習できません。',
      notOnIdolCardPage: 'アイドルカード育成ページにいません。PRODUCT_CARD_SELECTED が検出されませんでした。',
      externalPageRecoveryFailed: '外部ページ復旧失敗：ホームに戻り Produce を開きましたが、未完了プロデュース再開ダイアログに一致しませんでした。',
      produceFinishTimeout: 'プロデュース終了処理失敗：結果チェーンの進行がタイムアウトし、ホームに戻りませんでした。',
      consecutiveUnknownsExceeded: 'プロデュースメインループ: 連続画面認識不能の閾値を超えました。',
      noHandlerPause: 'プロデュースメインループ: 現在のページに使用可能なハンドラがなく、分析のため一時停止しました (phase={phase}, position={position})',
      maxGameplayLoopsExceeded: 'プロデュースメインループ: 最大ループ回数 {max_loops} に達しました。',
      supportCardSelectionFailed: 'サポートカード選択を完了できませんでした。',
      unknownSupportCardMode: '不明なサポートカード編成モード: {mode}',
      idolCardInfoNotFound: 'アイドルカード情報を取得できません。',
      trainingInfoCloseButtonNotFound: '育成情報閉じるボタンが見つからず、オーバーレイ閉じ点も推導できませんでした。',
      candidateListEmpty: '候補リストが空です。',
      combatPowerAnchorNotFound: '[総合力合計]アンカーポイントが見つかりません。',
    },
    gameAsset: {
      downloading: '{label} をダウンロードしています...',
      fetchingManifest: 'リソースマニフェストを取得しています...',
      fetchingManifestWithLabel: 'リソースマニフェストを取得しています（{label}）...',
      searchingWithLabel: '{label} のリソースを検索しています...',
      noMatchingResources: '{label} のリソースが見つかりませんでした',
      noMatchingObjects: '一致するオブジェクトが見つかりませんでした',
      downloadCompleted: '{label} のダウンロードが完了しました: 新規 {downloaded} 枚、スキップ {skipped} 枚（キャッシュ合計 {total} 枚）',
      phaseCompleted: '{label}: 新規 {downloaded} 枚、スキップ {skipped} 枚（キャッシュ合計 {total} 枚）',
      downloadFailed: '{label} のダウンロードに失敗しました',
      downloadFailedError: '{error}',
      dialogAssetsCompleted: 'サポートカード関連リソースのダウンロードがすべて完了しました',
      bulkDownloadFailed: 'リソースのダウンロードに失敗しました',
    },
    message: {
      gameNotForegroundRetry: 'ゲームが前面にありません。ゲーム画面に戻してから再試行してください。',
      gameNotStartedRetry: 'ゲームが起動していません。手動で起動してから再試行してください。',
      gameNotForegroundStart: 'ゲームが前面にありません。手動で起動してから再試行してください。',
      startLearningIdolClip: 'アイドルカード CLIP 学習を開始します。下部カルーセルを順番に走査します。',
    },
    config: {
      section: {
        base: '基本設定',
        dmm_player: 'DMM Player',
      },
      base: {
        run_mode: {
          label: '実行モード',
          hint: 'スクリプトの実行モードです（変更後は再起動が必要です）',
          option: {
            pc: 'PC（DMM）',
            phone: 'スマートフォン',
            mac_play_tools: 'macOS PlayCover',
          },
          disabledReason: {
            pcWindowsOnly: 'PC / DMM モードは Windows のみ対応しています。',
            macOnly: 'MacPlayTools モードは macOS（Apple Silicon）のみ対応しています。',
          },
        },
        ocr_backend: {
          label: 'OCR バックエンド',
          hint: 'auto: macOS では Vision を優先し、それ以外では RapidOCR を使用します。失敗時は RapidOCR にフォールバックします（変更後は再起動が必要です）',
          option: {
            auto: '自動',
            rapidocr: 'RapidOCR',
            vision: 'Vision（macOS 標準 OCR）',
          },
          disabledReason: {
            visionMacOnly: 'Vision OCR は macOS でのみ利用できます。',
          },
        },
        adb_connect_mode: {
          label: 'ADB 接続モード',
          hint: 'Android Debug Bridge の接続モードです。実機は USB、エミュレーターはネットワーク接続を推奨します（変更後は再起動が必要です）。',
          option: {
            network: {
              title: 'ネットワーク接続',
            },
            u_s_b: {
              title: 'USB 接続',
            },
          },
        },
        adb_host: {
          label: 'ADB ホスト名',
          hint: 'Android Debug Bridge の IP アドレスです。エミュレーターでは通常 127.0.0.1 を使用します。',
        },
        adb_port: {
          label: 'ADB ポート',
          hint: 'Android Debug Bridge のポートです。既定値は 5555 で、Android 11 以降ではシステムがランダムなポートを割り当てる場合があります。',
        },
        adb_serial: {
          label: 'USB 接続の ADB デバイス',
          hint: 'USB で接続したデバイスを選択してください。見つからない場合は一覧を更新してください。',
        },
        android_screen_capture_service: {
          label: 'ADB スクリーンショット方式',
          hint: 'scrcpy / DroidCast は通常 ADB スクリーンショットより低遅延です。scrcpy を使う場合は、公式 Releases の scrcpy-server を bin に配置してください。',
        },
        android_touch_service: {
          label: 'ADB タップ方式',
          hint: 'MaaTouch / minitouch / scrcpy を選択できます。MaaTouch は公式ビルド成果物を bin/maatouch に配置するか workflow で生成してください。minitouch は公式ビルド成果物を bin/minitouch に配置し、現在は Android 9 以下のみ対応です。',
        },
        auto_start_game: {
          label: 'ゲームを自動起動',
          hint: 'ゲームが起動していない場合に自動で起動します。',
        },
        auto_startup_time: {
          label: '自動実行時刻',
          hint: '24 時間表記、形式は HH:MM です。',
        },
        battle_decision_backend: {
          label: 'バトル判断バックエンド',
          hint: 'レッスン / 試験（Lesson / Exam）フェーズでのカード出し判断方式です。',
          option: {
            algo: {
              title: '簡易アルゴリズム',
            },
          },
        },
        check_resource_updates_on_startup: {
          label: '起動時にリソース更新を確認',
          hint: '起動後すぐに一度リソースリポジトリの更新確認を行います。',
        },
        disabled_tasks: {
          label: '無効化タスクリスト',
          hint: '無効化するタスクの一覧を設定します。',
        },
        enable_game_asset_download: {
          label: 'ゲームアセットのダウンロードを有効化',
          hint: 'GkmasObjectManager を使ってゲームサーバーからゲームアセットをダウンロードします。インターネット接続が必要です。',
        },
        enabled_auto_startup: {
          label: '毎日自動でスクリプトを実行',
          hint: '有効にすると、設定した時刻にタスクキューを自動で開始します。',
        },
        enabled_check_resource_updates: {
          label: '定期的にリソース更新を確認',
          hint: '設定した周期で assets/GakumasTranslationData と assets/gakumasu-diff の上流更新を確認します。',
        },
        gakumas_translation_data_repository_url: {
          label: 'GakumasTranslationData リポジトリ URL',
          hint: 'リソースのダウンロードと更新に使うリポジトリ URL です。変更すると即座に更新状態を再確認します。内容が分からない場合は変更しないでください。',
        },
        gakumasu_diff_repository_url: {
          label: 'gakumasu-diff リポジトリ URL',
          hint: 'リソースのダウンロードと更新に使うリポジトリ URL です。変更すると即座に更新状態を再確認します。内容が分からない場合は変更しないでください。',
        },
        game_package_name: {
          label: 'ゲームのパッケージ名',
          hint: '既定値: com.bandainamcoent.idolmaster_gakuen（変更後は再起動が必要です）',
        },
        game_window_name: {
          label: 'ゲームウィンドウ名',
          hint: '既定値: gakumas（変更後は再起動が必要です）',
        },
        llm_api_key: {
          label: 'LLM API Key',
          hint: 'API キーです。',
        },
        llm_base_url: {
          label: 'LLM API URL',
          hint: 'OpenAI 互換 API エンドポイント（llama / vLLM / OpenAI など）です。',
        },
        llm_insight_api_key: {
          label: '洞察モデル API Key',
          hint: '空欄の場合はメイン LLM Key を使用します。',
        },
        llm_insight_base_url: {
          label: '洞察モデル API URL',
          hint: '空欄の場合はメイン LLM の URL を使用します。独立したモデル（クラウド / 小型モデル）も指定できます。',
        },
        llm_insight_enabled: {
          label: '戦略洞察を有効化',
          hint: '後続判断に使う再利用可能な戦略洞察をバックグラウンドで生成します。',
        },
        llm_insight_max_tokens: {
          label: '洞察モデルの最大出力 Token',
          hint: '0 = 無制限。thinking + output の配分はモデルに任せます。',
        },
        llm_insight_model: {
          label: '洞察モデル',
          hint: '空欄の場合はメイン LLM モデル名を使用します。',
        },
        llm_insight_num_ctx: {
          label: '洞察モデルのコンテキスト長',
          hint: '0 = 未設定。API に自動管理させます。',
        },
        llm_insight_reasoning_effort: {
          label: '洞察モデルの思考強度',
          hint: '洞察生成時の推論深度を制御します。',
        },
        llm_insight_temperature: {
          label: '洞察モデルの温度',
          hint: '洞察生成時の温度です。低いほど結果が安定します（0.0 ~ 1.0）。',
        },
        llm_insight_timeout: {
          label: '洞察モデルのタイムアウト（秒）',
          hint: 'バックグラウンドの洞察生成タイムアウトで、メイン判断より長めに設定できます。',
        },
        llm_max_tokens: {
          label: 'LLM 最大出力 Token',
          hint: 'thinking + 回答を含む出力 token 上限です。0 にすると自動（API に送信しません）になります。',
        },
        llm_model: {
          label: 'LLM モデル',
          hint: 'モデル名（例: gpt-oss:20b、qwen3:4b、qwen3.5:9b など）です。',
        },
        llm_num_ctx: {
          label: 'LLM コンテキスト長',
          hint: '主に Ollama / ローカル OpenAI 互換バックエンド向けの任意互換パラメータです。0 で自動になります。',
        },
        llm_reasoning_effort: {
          label: 'LLM 思考強度',
          hint: '推論深度を制御します。low = 速い、medium = バランス、high = より十分に考えます。',
        },
        llm_temperature: {
          label: 'LLM 温度',
          hint: '生成温度です。低いほど結果が安定します（0.0 ~ 1.0）。',
        },
        llm_timeout: {
          label: 'LLM タイムアウト（秒）',
          hint: 'API リクエストのタイムアウトです。',
        },
        other_decision_backend: {
          label: 'その他の判断バックエンド',
          hint: '会話 / P ドリンク / スキル報酬 / 相談 / アイテム選択などのフェーズで使う判断方式です。',
          option: {
            algo: {
              title: '簡易アルゴリズム',
            },
          },
        },
        playtools_port: {
          label: 'PlayTools ポート',
          hint: 'PlayCover のゲームウィンドウタイトルバー [localhost:ポート番号] に表示されるポート番号です（変更後は再起動が必要です）。',
        },
        prefer_game_asset_image: {
          label: 'UI で常にゲームアセット画像を優先',
          hint: '有効にすると、UI 内のアイテム / サポートカード / スキルカード画像は、ゲーム中のスクリーンショットではなく、ゲームサーバーから取得した画像を優先して表示します。',
        },
        resource_update_check_period: {
          label: 'リソース更新確認周期',
          hint: '定期確認にのみ使用されます。',
          option: {
            daily: {
              title: '毎日',
            },
            every_3_days: {
              title: '3 日ごと',
            },
            weekly: {
              title: '毎週',
            },
          },
        },
        rl_inference_base_url: {
          label: 'RL 推論サービス URL',
          hint: 'ステートレスな RL 推論サービスの URL です。',
        },
        rl_inference_timeout: {
          label: 'RL 推論タイムアウト（秒）',
          hint: 'RL サービスへのリクエストタイムアウトです。',
        },
        schedule_decision_backend: {
          label: '週間行動判断バックエンド',
          hint: '週間行動（Schedule）フェーズでの自動判断方式です。',
          option: {
            algo: {
              title: '簡易アルゴリズム',
            },
          },
        },
      },
      dmm_player: {
        game_exe_path: {
          label: 'ゲームインストール先',
          hint: 'gakumas.exe を指すゲームのインストールパスです。通常は自動取得されるため変更不要です。',
        },
        viewer_id: {
          label: 'Viewer ID',
          hint: '自動取得されます。通常は変更不要です。',
        },
        open_id: {
          label: 'Open ID',
          hint: '自動取得されます。通常は変更不要です。',
        },
        pf_token: {
          label: 'PF Token',
          hint: '自動取得されます。通常は変更不要です。',
        },
      },
      task__auto_purchase: {
        weekly_gift: {
          label: '週次ギフトを購入',
          hint: 'ギフトページに無料購入できる項目があるか毎日確認します。',
        },
        daily_buy_list: {
          label: '毎日購入するアイテム',
          hint: '交換所で自動購入を許可するアイテムを選択します。',
        },
        refresh_shop: {
          label: '交換所を自動更新',
          hint: '交換所を毎日自動で更新します。',
        },
        use_gem_refresh: {
          label: 'ジュエルで交換所を更新',
          hint: '無料更新後も、ジュエルを使って更新を継続します。',
        },
      },
      task__auto_contest: {
        auto_reconfigure_team_before_challenge: {
          label: '挑戦前にチームを自動再編成',
          hint: 'チームに空き枠がある場合も自動編成を実行します。',
        },
        challenge_order: {
          label: '挑戦順',
          hint: '設定した順序に従って条件に合う挑戦相手を探します。',
          option: {
            random: {
              title: 'ランダム',
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
          label: '派遣時間を再設定',
          hint: '有効にすると、派遣前に勤務時間を再設定します。',
        },
        working_hours: {
          label: '派遣時間',
          hint: '「派遣時間を再設定」を有効にした場合のみ適用されます。',
          option: {
            '4_h': {
              title: '4 時間（最短）',
            },
            '8_h': {
              title: '8 時間',
            },
            '12_h': {
              title: '12 時間（最長）',
            },
          },
        },
      },
      task__auto_enhancement_support_card: {
        enhance_r: {
          label: 'R カードを強化',
          hint: 'R レアリティのサポートカードを自動で強化します。',
        },
        enhance_r_max_level: {
          label: 'R の最大強化レベル',
          hint: 'R カードの目標最大レベルです。',
        },
        enhance_sr: {
          label: 'SR カードを強化',
          hint: 'SR レアリティのサポートカードを自動で強化します。',
        },
        enhance_sr_max_level: {
          label: 'SR の最大強化レベル',
          hint: 'SR カードの目標最大レベルです。',
        },
        enhance_ssr: {
          label: 'SSR カードを強化',
          hint: 'SSR レアリティのサポートカードを自動で強化します。',
        },
        enhance_ssr_max_level: {
          label: 'SSR の最大強化レベル',
          hint: 'SSR カードの目標最大レベルです。',
        },
        auto_limit_break: {
          label: '上限解放を自動実行',
          hint: '同名カードがあり、星上限に達していない場合は自動で上限解放します。',
        },
        auto_convert: {
          label: '余剰サポートカードを自動変換',
          hint: '余剰の支援カードを自動で「サポートの証」に変換します。',
        },
        whitelist_mode: {
          label: 'ホワイトリストモード',
          hint: 'ホワイトリストで選択したカードのみ強化します。',
        },
        whitelist_card_ids: {
          label: 'ホワイトリストカード',
          hint: '自動強化を許可するサポートカードを選択します。',
        },
        whitelist: {
          no_selection: 'ホワイトリストのサポートカードはまだ選択されていません',
          open_dialog: 'ホワイトリストカードを選択',
          dialog_title: 'サポートカードホワイトリスト',
          search_placeholder: 'サポートカード名を検索（中国語 / 日本語 / ID 対応）',
          downloading_images: 'サポートカード関連画像をダウンロードしています...',
          add_to_whitelist: 'ホワイトリストに追加',
          limit_only: '限定',
          trigger_rate: 'サポート発生率',
          filter: {
            rarity: 'レア度',
            type: 'タイプ',
            plan: 'プラン',
          },
          type: {
            stamina_short: '体',
            assist_short: '補',
          },
          plan: {
            plan1: '感性',
            plan2: '理論',
            plan3: '異常',
            common: '共通',
          },
          card_category: {
            active_skill: 'アクティブスキル',
            mental_skill: 'メンタルスキル',
            trouble: 'トラブルカード',
            free_skill: 'フリースキル',
            skill_card: 'スキルカード',
            p_item: 'P アイテム',
          },
          section: {
            support_ability: 'サポート能力',
            support_event: 'サポートイベント',
            attachments: '付属報酬',
          },
        },
      },
      task__auto_producer: {
        scenario: {
          label: 'シナリオ',
          hint: '育成シナリオを選択します。',
          option: {
            hajime: {
              title: '初',
            },
          },
        },
        difficulty: {
          label: '難易度',
          hint: '育成難易度を選択します。',
        },
        nia_difficulty: {
          label: 'NIA 難易度',
          hint: 'NIA シナリオの難易度を選択します。',
        },
        target_idol_card_id: {
          label: '目標アイドルカード',
          hint: '対象 P アイドル ID です。空欄なら既定の選択カードを使います。先に「アイドルカード保存を更新」を実行してカード特徴を学習してください。',
        },
        support_card_mode: {
          label: 'サポートカード編成',
          hint: '自動編成またはプリセット番号を使用します。',
          option: {
            auto: {
              title: '自動編成',
            },
            preset: {
              title: 'プリセット番号',
            },
          },
        },
        support_card_preset_index: {
          label: 'サポートカードのプリセット番号',
          hint: '使用するサポートカード編成プリセットです。',
        },
        memory_mode: {
          label: 'メモリー編成',
          hint: '自動編成またはプリセット番号を使用します。',
          option: {
            auto: {
              title: '自動編成',
            },
            preset: {
              title: 'プリセット番号',
            },
          },
        },
        memory_preset_index: {
          label: 'メモリーのプリセット番号',
          hint: '使用するメモリー編成プリセットです。',
        },
        use_rental: {
          label: 'レンタルメモリーを使用',
          hint: 'メモリーを自動編成する際、「レンタルを使用」チェックボックスを有効にします。',
        },
        use_boost_items: {
          label: 'ブーストアイテムを使用',
          hint: '「開始確認」ページでブーストアイテムを使うかどうかです。',
        },
        resume_interrupted: {
          label: '中断した育成を再開',
          hint: '前回中断した育成を検出した場合、破棄して最初から始めるのではなく、自動で再開します。',
        },
        allow_ap_recovery: {
          label: 'AP 回復アイテムの使用を許可',
          hint: 'AP 不足時にアイテムで自動回復するかどうかです。',
        },
        allow_destroy_production_data: {
          label: '他端末の未完了育成データ破棄を許可',
          hint: '「プロデュースデータの破棄」が表示されたとき、確認して続行するかどうかです。',
        },
        schedule_notebook_mode: {
          label: 'P 手帳の読取戦略',
          hint: 'disabled: 読まない / before_decision: 週間行動の自動判断前のみ読む',
          option: {
            disabled: {
              title: '読取無効',
            },
            before_decision: {
              title: '判断前のみ読取',
            },
          },
        },
        memory_photo_mode: {
          label: 'メモリーフォト選択',
          hint: '育成終了後にメモリーフォトを選ぶ方法です。',
          option: {
            first: {
              title: '常に最初を選択',
            },
            vl: {
              title: 'VL 視覚モデルで最適なフォトを選択',
            },
          },
        },
        memory_photo_vl_prompt: {
          label: 'VL フォト選択プロンプト',
          hint: 'VL モデルがメモリーフォトを選ぶ際に使うカスタムプロンプトです。空欄なら既定のプロンプトを使用します。',
        },
        idol_card_browser: {
          no_selection: '目標アイドルカードはまだ選択されていません',
          search_placeholder: 'アイドルカード名を検索（中国語 / 日本語 / ID 対応）',
          limit_only: '限定',
          total: '合計',
          stamina: '体力',
          after_training: '育成後',
          plan: {
            plan1: '感性',
            plan2: '理論',
            plan3: '異常',
            common: '共通',
          },
          attribute: {
            vocal: 'ボーカル',
            dance: 'ダンス',
            visual: 'ビジュアル',
          },
          exam_effect: {
            parameter_buff: 'パラメータ上昇',
            review: '復習',
            lesson_buff: 'レッスン補正',
            concentration: '集中',
            card_play_aggressive: '攻撃的なカード運用',
            full_power: '全力',
          },
          filter: {
            rarity: 'レア度',
            plan: 'プラン',
            attribute: '属性',
            exam_effect: '試験効果',
            character: 'キャラクター',
          },
          section: {
            growth: '成長ステータス',
            skill_card: 'スキルカード',
            item: 'アイテム',
          },
          skin: {
            before: '覚醒前',
            after: '覚醒後',
          },
        },
      },
    },
    adb: {
      missing: 'adb がインストールされていません。Android SDK Platform-Tools をインストールし、adb を PATH に追加してください。',
      noError: '',
      notConnected: '現在 ADB デバイスは接続されていません。',
      invalidConnectMode: '無効な ADB 接続モードです: {mode}',
      deviceDisconnectedTarget: 'ADB デバイス {target} が切断されているか、接続されていません。エミュレーター/実機が起動中か確認してください。',
      deviceDisconnectedSerial: 'ADB デバイス {serial} が切断されているか、接続されていません。USB 接続、USB デバッグ有効化、WebUI でのデバイス一覧更新を確認してください。',
      deviceDisconnected: 'ADB デバイスが切断されているか、接続されていません。デバイスがオンラインか確認して再試行してください。',
      deviceOfflineTarget: 'ADB デバイス {target} は現在オフラインです。再試行するかデバイスを再起動してください。',
      deviceOfflineSerial: 'ADB デバイス {serial} は現在オフラインです。再接続し、USB デバッグの認可を確認してください。',
      deviceOffline: 'ADB デバイスは現在オフラインです。再接続してから再試行してください。',
      usbNotFoundSerial: '選択した USB ADB デバイスが見つかりません: {serial}。デバイス接続、USB デバッグ有効化、WebUI の一覧更新後に再選択してください。',
      usbNotFound: '利用可能な USB ADB デバイスが見つかりません。デバイスを接続し、USB デバッグを有効にして、WebUI で一覧を更新してください。',
      networkUnavailable: 'ADB デバイス {target} に接続できませんでした。adb のインストール、ネットワーク接続、adb tcpip / adb connect の実行を確認してください。元のエラー: {message}',
      initFailed: 'ADB 初期化に失敗しました: {message}',
      unavailable: 'Android デバイスは利用できません。',
      touchServiceUnavailable: 'タッチ操作に失敗しました: タッチサービスが利用できません。',
      reconnectFailed: 'ADB デバイスへの再接続に失敗しました。',
    },
    gameUtils: {
      modalHeaderTopEmpty: 'モーダルヘッダー位置を読み取れません: modal が空です',
      modalMissingHeaderBox: 'モーダル \'{title}\' に header_box がありません。安定性を判断できません',
      modalHeaderMissingCoords: 'モーダル \'{title}\' の header_box に y/cy 座標がありません。安定性を判断できません',
      modalSignatureEmpty: 'モーダル署名を構築できません: modal が空です',
      modalSignatureMissingHeader: 'モーダル \'{title}\' に header_box がありません。署名を構築できません',
      modalHeaderNoCenter: 'モーダル \'{title}\' の header_box の中心点を読み取れません',
      modalNoActionButtons: 'モーダル \'{title}\' に使用可能なアクションボタンがありません。署名を構築できません',
    },
  },
  config: {
    loadingHint: '長時間表示されない場合は、設定更新を反映するためバックエンドサービスを再起動してください。',
    unavailableOptionsPrefix: '現在選択できません: {items}',
    optionDisabledItem: '{title}: {reason}',
    componentNotRegistered: '未登録の設定コンポーネント: {component}',
  },
  logger: {
    close: 'ログを閉じる',
    open: 'ログを開く',
    title: '実行ログ',
  },
  api: {
    errorPrefix: 'API エラー: {message}{status}',
    statusSuffix: ' (status:{status})',
  },
})

export default ja
