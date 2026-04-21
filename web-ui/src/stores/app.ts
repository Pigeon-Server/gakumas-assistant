// Utilities
import { defineStore } from 'pinia'
import { toRef } from 'vue'
import apis from '@/scripts/apis'
import {TaskStatus, WS_ACTION} from '@/scripts/constants'
import { wsService } from "@/scripts/utils/websocket";
import message from "@/scripts/utils/message";
import dialogs from "@/scripts/utils/dialogs.js";
import {AppStatus, DeviceStatus} from "@/scripts/entity/status";
import {TaskItem} from "@/scripts/entity/task";
import {ConfigItem} from "@/scripts/entity/config";
import {ResourceUpdateStatus} from "@/scripts/entity/resourceUpdate";

/** Store State */
export interface AppState {
  status: AppStatus
  task_list: Record<string, TaskItem>
  config: Record<string, Record<string, ConfigItem>>
  resource_update_status: ResourceUpdateStatus | null
  resource_update_latest_event: string
  resource_update_latest_event_type: 'success' | 'warning' | 'info'
  last_prompted_resource_update_signature: string
  resource_update_prompt_open: boolean
  resource_bootstrap_prompt_open: boolean
  resource_bootstrap_prompt_dismissed: boolean
  resource_update_request_pending: boolean
}

export const useAppStore = defineStore('app', {
  state: (): AppState => ({
      status: {
        platform: '',
        yolo: false,
        task: TaskStatus.PENDING,
        current_task: "",
        suspended_task: "",
        device: {
          available: false,
          code: "initializing",
          message: "正在初始化设备...",
        },
        game: {
          current_location: '',
          player: {
          level: 0,
          gem: 0,
          stamina: 0
        }
      }
    },
    task_list: {},
    config: {},
    resource_update_status: null,
    resource_update_latest_event: "",
    resource_update_latest_event_type: "info",
    last_prompted_resource_update_signature: "",
    resource_update_prompt_open: false,
    resource_bootstrap_prompt_open: false,
    resource_bootstrap_prompt_dismissed: false,
    resource_update_request_pending: false,
  }),
  actions: {
    async init() {
      wsService.on(WS_ACTION.TaskStatusUpdate, (data) => {
        const task: TaskItem = this.get_task_by_id(data.id)
        if (!task) {
          return
        }
        console.log(`Update task '${data.id}' status: ${task.status} -> ${data.target_status}`)
        task.status = data.target_status
      })
      wsService.on(WS_ACTION.ResourceUpdateStatusChanged, (data) => {
        this.handle_resource_update_status(data)
      })
      wsService.on(WS_ACTION.DeviceStatusChanged, (data) => {
        this.apply_device_status(data)
      })
      wsService.on(WS_ACTION.AppStatusChanged, (data) => {
        this.apply_app_status(data)
      })
      const refreshBootstrap = async () => {
        try {
          await this.refresh_bootstrap_data()
        } catch (err) {
          console.debug("refresh_bootstrap_data failed", err)
        }
      }
      wsService.onEvent("connect", async () => {
        await refreshBootstrap()
      })
      wsService.onEvent("reconnect", async () => {
        await refreshBootstrap()
      })
    },
    async refresh_bootstrap_data() {
      await this.refresh_task_list()
      await this.refresh_app_status()
      await this.load_config()
      await this.refresh_resource_update_status()
    },
    async refresh_task_list() {
      const response = await apis.get_registered_tasks()
      this.task_list = response.data
    },
    async refresh_app_status() {
      const response = await apis.get_status()
      this.apply_app_status(response.data)
    },
    async run_task(task_name: string) {
      const task = this.get_task_by_id(task_name)
      const taskLabel = task?.description || task_name
      await apis.run_task(task_name)
      this.status.task = TaskStatus.RUNNING
      this.status.current_task = task_name
      this.status.suspended_task = ""
      if (task) {
        task.status = TaskStatus.RUNNING
      }
      message.showInfo(`已开始手动执行任务：${taskLabel}`)
    },
    async run_task_from(task_name: string) {
      const task = this.get_task_by_id(task_name)
      const taskLabel = task?.description || task_name
      await apis.run_task_from(task_name)
      this.status.task = TaskStatus.RUNNING
      this.status.current_task = task_name
      this.status.suspended_task = ""
      if (task) {
        task.status = TaskStatus.RUNNING
      }
      message.showInfo(`已从这里开始执行后续任务：${taskLabel}`)
    },
    async enable_task(task_name: string) {
      const task = this.get_task_by_id(task_name)
      const taskLabel = task?.description || task_name
      await apis.enable_task(task_name)
      if (task) {
        task.enable = true
      }
      await this.refresh_task_list()
      message.showSuccess(`已启用任务：${taskLabel}`)
    },
    async disable_task(task_name: string) {
      const task = this.get_task_by_id(task_name)
      const taskLabel = task?.description || task_name
      await apis.disable_task(task_name)
      if (task) {
        task.enable = false
      }
      await this.refresh_task_list()
      message.showSuccess(`已禁用任务：${taskLabel}`)
    },
    async load_config() {
      const response = await apis.get_config()
      this.config = response.data
    },
    async refresh_resource_update_status() {
      const response = await apis.get_resource_update_status()
      this.handle_resource_update_status(response.data)
    },
    async save_config() {
      await apis.save_config(this.config).then((response) => {
        this.config = response.data
        message.showSuccess("设置保存成功")
      })
    },
    notify_device_status_change(previousDevice?: DeviceStatus, currentDevice?: DeviceStatus) {
      if (!previousDevice || !currentDevice) {
        return
      }
      if (
        previousDevice.available === currentDevice.available
        && previousDevice.code === currentDevice.code
        && previousDevice.message === currentDevice.message
      ) {
        return
      }
      if (previousDevice.code === "initializing") {
        return
      }
      if (!previousDevice.available && currentDevice.available) {
        message.showSuccess("已自动识别到可用设备")
        return
      }
      if (previousDevice.available && !currentDevice.available) {
        message.showWarning(currentDevice.message || "设备连接已断开")
      }
    },
    apply_app_status(status: AppStatus) {
      const previousDevice = this.status?.device
      this.status = status
      this.notify_device_status_change(previousDevice, status?.device)
    },
    apply_device_status(deviceStatus: DeviceStatus) {
      const previousDevice = this.status?.device
      this.status.device = deviceStatus
      this.notify_device_status_change(previousDevice, deviceStatus)
    },
    format_bytes(value: number) {
      if (!Number.isFinite(value) || value <= 0) {
        return "0 B"
      }
      const units = ["B", "KB", "MB", "GB"]
      let size = value
      let unitIndex = 0
      while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024
        unitIndex += 1
      }
      const precision = size >= 100 || unitIndex === 0 ? 0 : 1
      return `${size.toFixed(precision)} ${units[unitIndex]}`
    },
    build_resource_progress_text(status: ResourceUpdateStatus | null) {
      const progress = status?.progress
      if (!progress?.active) {
        return ""
      }
      const parts: string[] = []
      if (progress.current_step && progress.total_steps) {
        parts.push(`步骤 ${progress.current_step}/${progress.total_steps}`)
      }
      if (progress.repository) {
        parts.push(progress.repository)
      }
      if (progress.bytes_total > 0) {
        parts.push(`${this.format_bytes(progress.bytes_downloaded)} / ${this.format_bytes(progress.bytes_total)}`)
      } else if (progress.percent > 0) {
        parts.push(`${progress.percent.toFixed(1)}%`)
      }
      if (progress.attempt > 0) {
        parts.push(`尝试 ${progress.attempt}/${progress.max_attempts}`)
      }
      if (progress.retry_wait_seconds > 0) {
        parts.push(`${progress.retry_wait_seconds}s 后重试`)
      }
      return parts.join(" / ")
    },
    handle_resource_update_status(status: ResourceUpdateStatus) {
      const previousStatus = this.resource_update_status
      this.resource_update_status = status
      if (status.updating || !status.bootstrap_required || status.required_resources_ready) {
        this.resource_update_request_pending = false
      }
      if (status.required_resources_ready) {
        this.resource_bootstrap_prompt_dismissed = false
      }
      if (status.checking && !status.updating) {
        this.resource_update_latest_event = ""
        this.resource_update_latest_event_type = "info"
      } else if (status.progress?.active && status.progress?.message) {
        this.resource_update_latest_event = status.progress.message
        this.resource_update_latest_event_type = "info"
      }
      if (previousStatus?.checking && !status.checking && !previousStatus?.updating) {
        if (status.last_error) {
          this.resource_update_latest_event = `资源检查失败：${status.last_error}`
          this.resource_update_latest_event_type = "warning"
        } else if (status.has_update) {
          this.resource_update_latest_event = "检测到资源仓库更新，可立即更新"
          this.resource_update_latest_event_type = "info"
        } else {
          this.resource_update_latest_event = "资源仓库已经是最新版本"
          this.resource_update_latest_event_type = "success"
        }
      }
      if (previousStatus?.updating && !status.updating) {
        if (status.last_error) {
          this.resource_update_latest_event = `资源更新失败：${status.last_error}`
          this.resource_update_latest_event_type = "warning"
        } else {
          const successMessage = previousStatus?.bootstrap_required || !previousStatus?.required_resources_ready
            ? "首次启动所需资源下载完成，游戏数据库已重新加载"
            : "资源仓库更新完成，游戏数据库已重新加载"
          this.resource_update_latest_event = successMessage
          this.resource_update_latest_event_type = "success"
          message.showSuccess(successMessage)
        }
      }
      this.maybe_prompt_resource_update(status)
    },
    build_resource_update_status_text(status: ResourceUpdateStatus | null) {
      if (!status) {
        return "尚未执行资源仓库检查"
      }
      const parts: string[] = []
      if (status.bootstrap_required && !status.required_resources_ready) {
        parts.push("首次启动需要下载游戏数据库和本地化资源")
      } else if (status.checking) {
        parts.push("正在检查资源仓库更新")
      } else if (status.updating) {
        parts.push("正在更新资源仓库")
      } else if (status.last_checked_at) {
        parts.push(`上次检查：${status.last_checked_at.replace("T", " ")}`)
      } else {
        parts.push("尚未执行资源仓库检查")
      }
      const progressText = this.build_resource_progress_text(status)
      if (progressText) {
        parts.push(progressText)
      }
      if (status.next_check_at) {
        parts.push(`下次定时检查：${status.next_check_at.replace("T", " ")}`)
      }
      return parts.join(" / ")
    },
    build_required_resource_prompt(status: ResourceUpdateStatus) {
      const repositories = status.missing_required_resources
        .map(item => `${item.name}（缺少 ${item.missing_count}/${item.required_count} 个文件）`)
      if (!repositories.length) {
        return "首次启动需要下载游戏数据库和本地化资源。下载过程中会显示进度，失败会自动重试。是否现在开始下载？"
      }
      return `首次启动需要下载以下资源：${repositories.join("、")}。下载过程中会显示进度，失败会自动重试。是否现在开始下载？`
    },
    build_resource_update_prompt(status: ResourceUpdateStatus) {
      const repositories = status.repositories
        .filter(repo => repo.has_update && !repo.error)
        .map(repo => `${repo.name}（${repo.local_commit_short} -> ${repo.remote_commit_short}）`)
      if (!repositories.length) {
        return "检测到资源仓库有更新，是否现在更新并重新加载游戏数据库？"
      }
      return `检测到资源仓库有更新：${repositories.join("、")}。是否现在更新并重新加载游戏数据库？`
    },
    async maybe_prompt_resource_update(status: ResourceUpdateStatus | null) {
      if (status?.bootstrap_required) {
        return
      }
      if (!status?.has_update || status.updating || !status.update_signature) {
        return
      }
      if (this.resource_update_prompt_open) {
        return
      }
      if (this.last_prompted_resource_update_signature === status.update_signature) {
        return
      }
      this.resource_update_prompt_open = true
      this.last_prompted_resource_update_signature = status.update_signature
      try {
        await dialogs.confirm(
          "发现资源仓库更新",
          this.build_resource_update_prompt(status),
          "立即更新",
          "稍后处理",
        )
        await this.apply_resource_updates()
      } catch (err) {
        this.last_prompted_resource_update_signature = ""
        console.debug("resource update prompt dismissed", err)
      } finally {
        this.resource_update_prompt_open = false
      }
    },
    dismiss_required_resource_download_prompt() {
      this.resource_bootstrap_prompt_dismissed = true
    },
    async start_required_resource_download() {
      this.resource_bootstrap_prompt_dismissed = true
      this.resource_update_request_pending = true
      await this.apply_resource_updates()
    },
    async check_resource_updates() {
      this.last_prompted_resource_update_signature = ""
      this.resource_update_latest_event = ""
      this.resource_update_latest_event_type = "info"
      await apis.check_resource_updates().then((response) => {
        this.handle_resource_update_status(response.data)
        if (response.data?.last_error) {
          this.resource_update_latest_event = response.message || response.data.last_error
          this.resource_update_latest_event_type = "warning"
          message.showWarning(response.message || response.data.last_error)
          return
        }
        this.resource_update_latest_event = response.message || "资源仓库检查完成"
        this.resource_update_latest_event_type = "success"
        if (!response.data?.has_update) {
          message.showSuccess(response.message || "资源仓库检查完成")
        }
      })
    },
    async apply_resource_updates() {
      const isBootstrap = Boolean(this.resource_update_status?.bootstrap_required || !this.resource_update_status?.required_resources_ready)
      this.resource_update_latest_event = isBootstrap ? "正在下载首次启动所需资源..." : "正在更新资源仓库..."
      this.resource_update_latest_event_type = "info"
      try {
        await apis.apply_resource_updates().then((response) => {
          this.handle_resource_update_status(response.data)
        })
      } finally {
        if (!this.resource_update_status?.updating) {
          this.resource_update_request_pending = false
        }
      }
    },
    get_task_config(task_name: string) {
      // console.log(`task__${task_name}`,this.config?.[`task__${task_name}`])
      return toRef(this.config, `task__${task_name}`)
    },
    async save_task_config(task_name: string) {
      await apis.save_task_config(task_name, this.config?.[`task__${task_name}`]).then((response) => {
        const target_config = `task__${task_name}`
        this.config[target_config] = response.data
        message.showSuccess("设置保存成功")
      })
    },
    async reset_config() {
      await apis.reset_config().then(response => {
        this.config = response.data
        message.showSuccess("设置重置完成，部分设置可能需要重启生效")
      })
    },
    get_task_by_id(task_id: string): TaskItem | undefined {
      const task: TaskItem = this.task_list?.[task_id]
      if (!task) {return}
      return task
    },
    get_current_task(): TaskItem | undefined {
      return this.get_task_by_id(this.status?.current_task || "")
    },
    get_suspended_task(): TaskItem | undefined {
      return this.get_task_by_id(this.status?.suspended_task || "")
    }
  }
})
