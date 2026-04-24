class ProduceText:
    """プロデュース（培育）相关的游戏内文本常量"""

    # 确认页 / 育成情报
    AFFINITY = "親愛度"  # 亲密度
    TRAINING_INFO = "育成情報"  # 育成信息
    RECOMMENDED_EFFECT = "おすすめ効果"  # 推荐效果
    FORMATION_DETAILS = "編成詳細"  # 编成详细

    # 参数名称
    VOCAL = "ボーカル"
    DANCE = "ダンス"
    VISUAL = "ビジュアル"
    STAMINA = "体力"
    VOCAL_OCR_VARIANTS = (VOCAL, "ポーカル", "ホーカル", "ボ一カル")
    DANCE_OCR_VARIANTS = (DANCE, "タンス")
    VISUAL_OCR_VARIANTS = (VISUAL, "ヒジュアル")
    STAMINA_OCR_VARIANTS = (STAMINA, "体カ")
    VOCAL_LESSON = "ボーカルレッスン"
    DANCE_LESSON = "ダンスレッスン"
    VISUAL_LESSON = "ビジュアルレッスン"
    PARAMETER = "パラメータ"
    PARAMETER_OCR_VARIANTS = (PARAMETER, "バラメータ")
    PARAMETER_UP = "パラメータ上昇"
    PARAMETER_UP_INCREASE = "パラメータ上昇量増加"
    INCREASE = "上昇"
    SCORE = "スコア"
    BLOCK = "ブロック"
    TURN = "ターン"
    NOT_MULTIPLE = "複数不可"
    CARD_TYPE_ACTIVE = "アクティブ"
    CARD_TYPE_MENTAL = "メンタル"
    CARD_TYPE_TROUBLE = "トラブル"

    # 战斗 / 效果关键字
    GOOD_IMPRESSION = "好印象"
    CONCENTRATION = "集中"
    GOOD_CONDITION = "好調"
    EXCELLENT_CONDITION = "絶好調"
    GENKI = "元気"
    YARUKI = "やる気"
    ENTHUSIASM = "熱意"
    FULL_POWER = "全力"
    FULL_POWER_POINT = "全力値"
    STRONG_SPIRIT = "強気"
    WEAK_SPIRIT = "弱気"
    CONSERVE_POWER = "温存"
    SKILL_CARD_USE_COUNT_UP = "スキルカード使用数追加"
    SKILL_CARD_USE_COUNT_UP_SHORT = "使用数追加"
    STAMINA_RECOVERY = "体力回復"
    STAMINA_CONSUMPTION = "消費体力"
    STATUS_VALUE_TOKENS = (
        GOOD_IMPRESSION,
        CONCENTRATION,
        GOOD_CONDITION,
        GENKI,
        ENTHUSIASM,
        FULL_POWER_POINT,
    )
    BATTLE_LOGIC_TOKENS = (
        GOOD_IMPRESSION,
        YARUKI,
        GENKI,
    )
    BATTLE_SENSE_TOKENS = (
        GOOD_CONDITION,
        EXCELLENT_CONDITION,
        CONCENTRATION,
    )
    BATTLE_ANOMALY_TOKENS = (
        FULL_POWER,
        FULL_POWER_POINT,
        STRONG_SPIRIT,
        CONSERVE_POWER,
        ENTHUSIASM,
    )
    BATTLE_SETUP_TOKENS = (
        GOOD_CONDITION,
        EXCELLENT_CONDITION,
        CONCENTRATION,
        GOOD_IMPRESSION,
        YARUKI,
        ENTHUSIASM,
        FULL_POWER_POINT,
        STRONG_SPIRIT,
        CONSERVE_POWER,
        PARAMETER_UP_INCREASE,
    )
    BATTLE_RECOVERY_TOKENS = (
        GENKI,
        STAMINA_RECOVERY,
        STAMINA_CONSUMPTION,
    )
    BATTLE_IMMEDIATE_OUTPUT_TOKENS = (
        SCORE,
        VOCAL,
        DANCE,
        VISUAL,
    )

    # 编成详细 - 子页签
    TAB_CARD_ITEM = "カード/アイテム"  # 卡片/道具
    TAB_ABILITY = "アビリティ"  # 能力
    TAB_EVENT = "イベント"  # 事件

    # 编成详细 - 能力分区标题
    LESSON_SUPPORT = "レッスンサポート"  # 课程支援
    SUPPORT_ABILITY = "サポートアビリティ"  # 支援能力
    MEMORY_ABILITY = "メモリーアビリティ"  # 回忆能力
    P_IDOL_ABILITY = "Pアイドルアビリティ"  # P偶像能力
    SKILL_CARD_SUPPORT = "スキルカードサポート"  # 技能卡支援

    # 编成详细 - 卡片来源标题
    OWNED_AT_START = "プロデュース開始時から所持"  # 开始时持有
    OWNED_AT_START_SHORT = "開始時から所持"  # 开始时持有（简称）
    EARNED_DURING_PRODUCE = "プロデュース中獲得"  # 育成中获得

    # 编成详细 - 噪声文本（OCR 过滤用）
    GUIDE = "獲得ガイド"  # 获得指南
    SKILL_CARD_SWITCH = "スキルカードスイッチ設定"  # 技能卡切换设置

    # 育成课题 - 任务类型
    TRAINING_TASKS = "育成課題"  # 育成课题
    TASK_TYPE_PERFORMANCE = "実力発揮"  # 实力发挥
    TASK_TYPE_WEAKNESS = "弱点克服"  # 弱点克服
    TASK_TYPE_PERFORMANCE_OCR_VARIANTS = (
        TASK_TYPE_PERFORMANCE,
        "実カ発準",
        "実カ発揮",
    )

    # 育成课题 - 比较运算符
    COMPARISON_GE = "以上"  # ≥
    COMPARISON_LE = "以下"  # ≤

    # 阶段关键词（判定文本是否为阶段描述）
    PHASE_KEYWORDS = ("獲得", "開始時", "終了時", "終了後", "試験", "審査", "オーディション")

    # 审查 / 试验关键词
    MID_EXAM = "中間試験"  # 中间考试
    MID_REVIEW = "中間審査"  # 中间审查
    FIRST_AUDITION = "1次オーディション"  # 第一次试镜

    # ── 培育ゲームプレイ中テキスト ──
    VOICE_PLAYBACK_CONFIRM = "ボイス再生確認"  # 语音播放确认
    COMMU_FAST_FORWARD = "コミュ早送り設定"  # 对话快进设置
    PRODUCE_SKIP_SETTINGS = "プロデュース演出スキップ設定"  # 培育演出跳过设置
    MESSAGE = "メッセージ"  # 剧情消息 / 来信标题常见文本
    STORY_DIALOGUE_RECOVERY_HINT_TOKENS = (
        "見ました",
        "話がある",
        "って",
    )  # 剧情过场页常见 OCR 片段
    RECEIVE = "受け取る"  # 奖励领取确认按钮
    RECEIVE_COMPLETE = "受取完了"  # 领取完成
    P_DRINK_SELECT = "受け取るPドリンクを選んでください"  # P饮料选择提示
    P_ITEM_SELECT = "受け取るPアイテムを選んでください"  # P物品选择提示
    SKILL_REWARD_SELECT = "受け取るスキルカードを選んでください"  # 技能卡奖励选择提示
    P_DRINK = "Pドリンク"  # P 饮料
    P_DRINK_DISCARD = "捨てる"  # P饮料详情模态内的丢弃按钮
    P_DRINK_DISCARD_CONFIRM = "廃棄確認"  # P饮料丢弃确认弹窗标题
    P_DRINK_DISCARD_CONFIRM_YES = "はい"  # P饮料丢弃确认 - 确认
    P_DRINK_DISCARD_CONFIRM_NO = "いいえ"  # P饮料丢弃确认 - 取消
    GAME_TITLE = "学園アイドルマスター"  # 游戏标题 Logo（异常回到标题/启动页时使用）
    DETAIL = "詳細"  # 详情
    SELECT_PROMPT = "選んでください"  # 通用“请选择”提示
    LEGEND = "レジェンド"  # Legend 难度
    DIFFICULTY_MASTER = "マスター"  # Master 难度
    DIFFICULTY_REGULAR = "レギュラー"  # Regular 难度
    DIFFICULTY_PRO = "プロ"  # Pro 难度
    DIFFICULTY_LABEL_MAP = {
        DIFFICULTY_MASTER: "master",
        DIFFICULTY_REGULAR: "regular",
        DIFFICULTY_PRO: "pro",
        LEGEND: "legend",
    }
    RENTAL = "レンタル"  # 租借
    WEEK = "週"  # 周
    MEMORY_FORMATION = "メモリー編成"  # 记忆编成
    MEMORY = "メモリー"  # メモリー
    MEMORY_OCR_VARIANTS = (MEMORY, "MEMORY")  # メモリー 的常见 OCR 形式
    OWNED_MEMORY = "所持メモリー"  # 所持记忆
    AVAILABLE_SKILL_CARD = "獲得可能スキルカード"  # 记忆详情中的可获得技能卡分页标题
    MEMORY_LIST = "メモリー一覧"  # 记忆列表
    MEMORY_CONVERT = "メモリー変換"  # 记忆转换
    PRODUCE_RESULT = "プロデュース結果"  # 培育结果
    PRODUCE = "プロデュース"  # 培育 / Produce
    PRODUCE_EVALUATION = "プロデュース評価"  # 培育评价
    PRODUCE_COMPLETE = "プロデュース完了"  # 培育完成
    PRODUCE_RESUME = "プロデュース再開"  # 继续未完成的培育弹窗标题
    PRODUCE_RETIRE_CONFIRM = "プロデュースリタイア確認"  # 放弃培育确认
    GAMEPLAY_MENU_SUSPEND = "中断"  # 局内菜单：保存并中断
    GAMEPLAY_MENU_HELP = "ヘルプ"  # 局内菜单：帮助
    GAMEPLAY_MENU_SETTINGS = "設定"  # 局内菜单：设置
    GAMEPLAY_MENU_RANKING = "ランキング"  # 局内菜单：排行榜
    FINAL_PRODUCE_EVALUATION = "最終プロデュース評価"  # 最终培育评价
    PASS = "合格"  # 合格
    REWARD_ITEMS = "獲得アイテム"  # 获得道具
    FINAL_EXAM = "最終試験"  # 最终考试
    FINAL_REVIEW = "最終審査"  # 最终审查
    EXAM_RESULT_RETRY_CONFIRM = "再挑戦確認"  # 再挑战确认
    END_TURN_CONFIRM = "ターン終了"  # 结束当前回合确认
    HAND = "手札"  # 手牌
    SKILL_CARD = "スキルカード"  # 技能卡
    ZERO_CARDS = "0枚"  # 0 张
    ZERO_CARDS_OCR_VARIANTS = (ZERO_CARDS, "０枚", "O枚")  # OCR 常见 0 枚误读
    EMPTY_HAND_MESSAGE = "手札のスキルカードが0枚です"  # 战斗中无手牌提示
    MEMORY_EFFECT = "メモリー効果"  # 记忆效果
    MEMORY_REGEN_CONFIRM = "メモリー再生成確認"  # 记忆再生成确认
    MEMORY_CONFIRM = "メモリー確定確認"  # 记忆确定确认
    MEMORY_GENERATION_COMPLETE = "メモリー生成完了"  # 记忆生成完成
    MEMORY_SELECT = "獲得するメモリーを選択してください"  # 选择要获得的记忆
    MEMORY_PHOTO_SELECT = "メモリーにするフォトを選んでください"  # 选择记忆卡面照片
    MEMORY_PHOTO_SELECT_SHORT = "フォトを選"  # 照片选择标题常见残缺 OCR
    MEMORY_PHOTO_SELECT_PREFIX = "メモリーにする"  # 记忆卡面标题前缀
    PRODUCE_HISTORY = "プロデュース履歴"  # 培育历史
    ACHIEVEMENT_PROGRESS = "アチーブメント進捗"  # 成就进度
    EVENT_REWARD_PROGRESS = "イベント報酬進捗"  # 事件奖励进度
    EVENT_POINT = "イベントPt"  # 事件点数
    EVENT = "イベント"  # 事件 / Event
    UNREAD_COMMU_FAST_FORWARD_CONFIRM = "未読のコミュです"  # 未读对话快进确认
    FAILED = "不合格"  # 不合格
    REWARD = "報酬"  # 奖励
    CONSULT = "相談"  # 相談行动
    ACTIVITY = "活動"  # 活动类事件
    PRESENT_SUPPORT = "活動支給"  # 活动支给
    PRESENT_SELECTION = "差し入れ選択時"  # 活动支给 / 差し入れ 选项说明
    PRESENT_SELECTION_SHORT = "選択時"  # OCR 可能只读到选项说明后半段
    FAN_PRESENT = "差し入れ"  # 差し入れ
    BUSINESS = "営業"  # 营业
    BUSINESS_CORPORATE = "企業イベント"  # 企业活动营业
    BUSINESS_MUNICIPAL = "自治体イベント"  # 自治体活动营业
    BUSINESS_RESORT = "リゾート施設"  # 度假设施营业
    BUSINESS_COMMERCIAL = "商業施設"  # 商业设施营业
    OUTING = "おでかけ"  # 外出
    GO_OUT = "外出"  # 外出（替代写法）
    CLASS = "授業"  # 授业
    CLASS_LESSON_VOCAL = "ボーカル通常レッスンを開始"  # 授業：ボーカル效果描述
    CLASS_LESSON_DANCE = "ダンス通常レッスンを開始"  # 授業：ダンス效果描述
    CLASS_LESSON_VISUAL = "ビジュアル通常レッスンを開始"  # 授業：ビジュアル效果描述
    REST = "休"  # 休息
    REST_ACTION = "休む"  # 休む
    SELF_LESSON = "自主"  # 自主训练
    HARD_LESSON = "追い込み"  # 追い込み训练
    LESSON = "レッスン"  # 课程 / レッスン
    AUDITION = "オーディション"  # 选秀 / 试镜
    SECOND_AUDITION = "2次オーディション"  # 第二次选秀
    FINALE = "FINALE"  # 最终舞台
    FES = "フェス"  # フェス
    LIVE = "ライブ"  # ライブ
    MID = "中間"  # 中间阶段关键词
    FINAL = "最終"  # 最终阶段关键词
    PLAN_SENSE = "センス"  # 流派：Sense
    PLAN_LOGIC = "ロジック"  # 流派：Logic
    PLAN_ANOMALY = "アノマリー"  # 流派：Anomaly
    SPECIAL_GUIDANCE = "特別指導"  # 特别指导
    CUSTOMIZE = "カスタマイズ"  # 卡牌自定义
    SKILL_CARD_REMOVE = "削除"  # 技能卡删除
    ENHANCE_CONFIRM = "強化する"  # 强化确认
    EXAM_CRITERIA = "審査基準"  # 审查基准
    FINAL_EXAM_CRITERIA_HEADING = "最終試験の審査基準"  # 育成信息中的审查基准标题
    PASS_CONDITION = "合格条件"  # 合格条件
    TAP_TO_CONTINUE = "タップして次へ"  # 点击继续
    REMAINING_TURNS = "残りターン"  # 剩余回合（考试轮盘上方标签）
    REMAINING_TURNS_OCR_VARIANTS = ("残りターン", "洗りターン", "残リターン")
    SKILL_CARD_DETAIL_HINT_TOKENS = (
        STAMINA,
        LESSON,
        SKILL_CARD,
        "ターン",
        PARAMETER,
        *PARAMETER_OCR_VARIANTS,
        GENKI,
        "複数不可",
        INCREASE,
        GOOD_IMPRESSION,
        MEMORY,
    )
    SKILL_REWARD_SHOWCASE_VERBS = (
        "強化しました",
        "獲得しました",
        "習得しました",
        "入手しました",
        "チェンジしました",  # 卡片更换通知（例：「…にチェンジしました」）
    )  # 单卡展示页常见结算动词
    RANKING_POINT_UNIT = "pt"  # 排行榜分数单位
    PASS_POSITION_SUFFIX = "位以上"  # 合格线常见后缀
    PRAISE_EXCELLENT = "すばらしい"  # 结果展示夸奖文案
    PRAISE_EXCELLENT_OCR_VARIANTS = (PRAISE_EXCELLENT, "すばら")
    RANKING_ORDINAL_MARKERS = ("1st", "2nd", "3rd", "4th", "5th", "6th")

    # ── 技能卡奖励 再抽選 ──
    REDRAW = "再抽選"  # 再抽选按钮
    REDRAW_REMAINING_KEYWORDS = ("あと", "回")  # 再抽选剩余次数识别关键词
    REMAINING_COUNT_PATTERN = r"あと\s*(\d+)\s*回"  # 常见“あとN回”提示
    REDRAW_REMAINING_DISPLAY_TEMPLATE = "（あと{remaining}回）"  # 再抽選按钮展示
    RECOMMEND = "おすすめ"  # 推荐徽章
    REDRAW_SHORT = "再"  # 再抽選按钮常见残缺 OCR

    # ── 周行動 (Schedule) 関連 ──
    SCHEDULE = "スケジュール"  # 日程安排选项卡
    SCHEDULE_CONFIRM = "スケジュール確認"  # 日程确认
    SCHEDULE_SELECT = "スケジュール選択"  # 日程选择
    SCHEDULE_NOTEBOOK = "Pノート"  # P手帳（日程规划笔记本）
    SCHEDULE_NOTEBOOK_ALT = "P手帳"  # P手帳（替代写法）
    ACTION_INFO_EFFECT = "効果"  # 行动效果标签
    ACTION_INFO_TRAINING = "トレーニング"  # 训练标签
    SCHEDULE_LOOKUP_NOISE_TOKENS = (
        PRESENT_SELECTION,
        PRESENT_SUPPORT,
        EXAM_CRITERIA,
        PARAMETER_UP,
    )
    NOTEBOOK_SPECIAL_KEYWORDS = (
        "試験",
        AUDITION,
        HARD_LESSON,
        LESSON,
        FES,
        LIVE,
        MID,
        FINAL,
    )
    NOTEBOOK_ICON_LABEL_MAP = {
        VISUAL_LESSON: "visual",
        DANCE_LESSON: "dance",
        VOCAL_LESSON: "vocal",
        OUTING: "outing",
        BUSINESS: "business",
        CLASS: "class",
        REST_ACTION: "refresh",
    }

    # ── AP 相关テキスト ──
    AP_SHORTAGE = "AP不足"  # AP 不足弹窗标题
    AP_RECOVERY = "AP回復"  # AP 回复弹窗标题
    AP_DRINK = "APドリンク"  # AP 回复道具名
    AP_DRINK_USE = "使う"  # 使用 AP 道具按钮
    AP_RECOVER_BUTTON = "回復する"  # AP 回复确认按钮
    AP_CANCEL = "キャンセル"  # 取消按钮

    # ── ライブ演出テキスト ──
    LANDSCAPE_START_NOTICE = "横画面で開始します"  # live 开始前的横屏提示
    LANDSCAPE_START_NOTICE_OCR_VARIANTS = (
        "横画面で開始します",
        "横画面で開始しま",
        "横画面で開始",
    )
    TAP_TO_START = "TAP TO START"  # ライブ演出開始
    TAP_TO_START_OCR_VARIANTS = ("TAP TO START", "TAPTO START", "TAP TOSTART", "TAPTOSTART")
    TAP = "TAP"  # 点击推进 / TAP
    SKIP = "SKIP"  # 剧情跳过 / Skip
    LOADING = "LOADING"  # Loading
    NOW_LOADING = "NOWLOADING"  # 去空格后的 NOW LOADING
    LOADING_TOKENS = (NOW_LOADING, LOADING)
    REWARD_RECEIVE_DETAIL_TOKENS = (
        YARUKI,
        GENKI,
        PARAMETER,
        SKILL_CARD,
    )  # 奖励领取确认页中常见的奖励说明词
    RESULT_RESTORED = "戻しました"  # 结果页撤回后的提示
    RESULT_HISTORY_DETAIL_TOKENS = (
        "定期公演",
        "編成",
        "PLV",
        "編成詳細",
        "サポートカード",
    )  # 结果页详情面板的识别词
    PHOTO_SAVE_TO_DEVICE = "端末保存"  # 保存到设备
    PHOTO_SAVE_TO_ALBUM = "アルバム保存"  # 保存到相册

    # ── Pドリンク詳細モーダル ──
    P_DRINK_DETAIL = "Pドリンク詳細"  # P饮料详情模态标题
    P_DRINK_USE = "使う"  # P饮料详情模态内的使用按钮
    P_DRINK_LIMIT_NO_RECEIVE = "受け取らない"  # 所持上限页：不领取新饮料
    P_DRINK_LIMIT_KEEP = "残す"  # 所持上限页：保留新饮料/确认文案之一
    DO_NOT_ACQUIRE = "獲得せず"  # 不获取
    P_DRINK_LIMIT_SKIP_ALTS = (
        P_DRINK_LIMIT_NO_RECEIVE,
        RECEIVE,
        P_DRINK_LIMIT_KEEP,
        DO_NOT_ACQUIRE,
    )
    POSSESSION_LIMIT = "所持上限"  # 所持上限
    POINT = "ポイント"  # ポイント
    P_POINT = "Pポイント"  # Pポイント
    DRINK = "ドリンク"  # ドリンク
    BONUS = "ボーナス"  # ボーナス
    TICKET = "チケット"  # チケット
    DIRECT_STAMINA_COST_HINT_TOKENS = (
        "元気は体力のかわりに消費できません",
        "元気のかわりに消費できません",
        "体力を直接消費",
        "体力直接消費",
    )
    ITEM_SELECT_NOISE_TOKENS = (
        P_ITEM_SELECT,
        SELECT_PROMPT,
        RECEIVE,
    )
    SKILL_CARD_USE_CONFIRM = "スキルカード使用確認"  # 技能卡使用确认弹窗标题
    EXECUTE_CONFIRM = "実行しますか"  # 执行确认提示
    NO_EFFECT_TRIGGERED = "効果が発動しません"  # 效果未发动提示
    ZERO_VALUE_NO_EFFECT_PREFIX = "の値が0のため"  # 零值导致效果不发动提示前缀

    # ── 課程/試験 進捗サークル ──
    # 课程画面中央的进度圆圈 OCR 文本关键词
    PROGRESS_PERFECT = "PERFECT"  # 进度圆圈: PERFECT 阶段
    PROGRESS_CLEAR = "CLEAR"  # 进度圆圈: CLEAR 阶段
    PROGRESS_MADE = "まで"  # 进度圆圈: "…まで" 后缀（距离目标）
    # OCR 抗噪变体
    PROGRESS_PERFECT_OCR_VARIANTS = (
        "PERFECT", "PERFEC7", "PERFECF", "PERFECI",
        "パーフェクト", "パーフエクト",
    )
    PROGRESS_CLEAR_OCR_VARIANTS = (
        "CLEAR", "C1EAR", "CIEAR",
        "クリア", "クリァ",
    )
    PROGRESS_MADE_OCR_VARIANTS = ("まで", "まて", "まで")  # まで → まて 是常见 OCR 误读
