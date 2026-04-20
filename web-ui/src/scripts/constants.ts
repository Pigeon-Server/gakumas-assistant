export const WS_ACTION_HEAD = 'action' as const

export const WS_ACTION = {
  AppStatusChanged: 'app:status_changed',
  TaskStatusUpdate: 'task:status_update',
  TaskQueueStart: 'task:start',
  TaskQueueStop: 'task:stop',
  TaskQueueSuspend: "task:suspend",
  UpdateCurrentTask: "task:update_current_task",
  TaskExecutionError: "task:execution_error",

  ShowMessage_Info: 'show_msg:info',
  ShowMessage_Warning: 'show_msg:warning',
  ShowMessage_Error: 'show_msg:error',
  ShowMessage_Success: 'show_msg:success',

  ResourceUpdateStatusChanged: 'resource_update:status_changed',
  DeviceStatusChanged: 'device:status_changed',

  BroadcastLog: 'broadcast_log',
  Ping: 'ping',
  Pong: 'pong',
} as const

export const TaskStatus = {
  PENDING: "PENDING",
  RUNNING: "RUNNING",
  SUCCESS: "SUCCESS",
  RETRY: "RETRY",
  FAILED: "FAILED",
  CANCELED: "CANCELED",
  SUSPENDED: "SUSPENDED"
} as const
