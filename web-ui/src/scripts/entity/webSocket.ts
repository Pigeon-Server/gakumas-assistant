import {WS_ACTION} from "@/scripts/constants.js"
import {TaskItem} from "@/scripts/entity/task";
import {AppStatus} from "@/scripts/entity/status";

export interface WsOptions {
  reconnect?: boolean
  reconnectInterval?: number      // 初始重连间隔
  maxReconnectInterval?: number   // 最大重连间隔
  heartbeatInterval?: number
}

export interface WsEventPayloads {
  [WS_ACTION.AppStatusChanged]: AppStatus
  [WS_ACTION.TaskStatusUpdate]: {
    id: string
    target_status: TaskItem['status']
  }
  [WS_ACTION.UpdateCurrentTask]: {
    task_id: string
  }
  [WS_ACTION.TaskQueueStart]: void
  [WS_ACTION.TaskQueueStop]: void
  [WS_ACTION.TaskQueueSuspend]: void
  [WS_ACTION.TaskExecutionError]: {
    task_id: string
    task_name: string
    status: string
    error_type: string
    error_message: string
    dump_dir: string
    package_path: string
    package_id: string
    package_download_url: string
    feedback: {
      github_issues: string
      qq_group: string
    }
  }
  [WS_ACTION.BroadcastLog]: {
    time: string
    level: string
    message: string
  }
}
