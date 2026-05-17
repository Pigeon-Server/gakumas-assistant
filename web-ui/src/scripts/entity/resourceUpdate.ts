import type { I18nText } from '@/scripts/i18n/types'

export interface ResourceRepositoryStatus {
  name: I18nText | string
  path: string
  exists: boolean
  dirty: boolean
  has_update: boolean
  local_commit: string
  remote_commit: string
  local_commit_short: string
  remote_commit_short: string
  error: string
}

export interface ResourceBootstrapMissingResource {
  name: I18nText | string
  path: string
  required_count: number
  missing_count: number
  missing_paths: string[]
}

export interface ResourceUpdateProgress {
  active: boolean
  phase: string
  title: I18nText | string
  message: I18nText | string
  repository: I18nText | string
  repository_path: string
  current_step: number
  total_steps: number
  step_percent: number
  percent: number
  bytes_downloaded: number
  bytes_total: number
  attempt: number
  max_attempts: number
  retry_wait_seconds: number
}

export interface ResourceUpdateStatus {
  enabled: boolean
  check_on_startup: boolean
  check_period: string
  interval_minutes: number
  checking: boolean
  updating: boolean
  has_update: boolean
  last_checked_at: string | null
  next_check_at: string | null
  last_error: I18nText | string
  update_signature: string
  required_resources_ready: boolean
  bootstrap_required: boolean
  missing_required_resources: ResourceBootstrapMissingResource[]
  progress: ResourceUpdateProgress
  repositories: ResourceRepositoryStatus[]
}
