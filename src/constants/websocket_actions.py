
class WebsocketActions:
    BaseActionFlag = "action"

    class App:
        StatusChanged = "app:status_changed"

    class WebsocketHeartBeat:
        Ping = "ping"
        Pong = "pong"

    class TaskService:
        TaskStatusUpdate = "task:status_update"
        UpdateCurrentTask = "task:update_current_task"
        TaskQueueStart = "task:start"
        TaskQueueStop = "task:stop"
        TaskQueueSuspend = "task:suspend"
        TaskExecutionError = "task:execution_error"

    class ResourceUpdate:
        StatusChanged = "resource_update:status_changed"

    class Device:
        StatusChanged = "device:status_changed"

    class Message:
        ShowMessage_Info = "show_msg:info"
        ShowMessage_Warning = "show_msg:warning"
        ShowMessage_Error = "show_msg:error"
        ShowMessage_Success = "show_msg:success"

    class Logger:
        BroadcastLog = "broadcast_log"
