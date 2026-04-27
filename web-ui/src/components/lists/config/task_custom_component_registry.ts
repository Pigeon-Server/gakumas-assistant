import SelectItem from '@/components/lists/config/task_settings/purchase_setting/select_item.vue'
import SupportCardWhitelist from '@/components/lists/config/task_settings/enhancement_setting/support_card_whitelist.vue'
import IdolCardBrowser from '@/components/lists/config/task_settings/auto_producer/idol_card_browser.vue'

export interface TaskConfigFieldContext {
  config: Record<string, any>
  section: Record<string, any>
  sectionName: string
  fieldName: string
  item: Record<string, any>
  save?: (() => void) | null
}

interface TaskCustomComponentEntry {
  component: any
  buildProps?: (context: TaskConfigFieldContext) => Record<string, any>
}

export const taskCustomComponentRegistry: Record<string, TaskCustomComponentEntry> = {
  task_auto_purchase_item_selector: {
    component: SelectItem,
    buildProps: ({ section, item }) => ({
      data: section,
      label: item?.ui?.label,
    }),
  },
  task_auto_enhancement_support_card_whitelist: {
    component: SupportCardWhitelist,
    buildProps: ({ section }) => ({
      data: section,
    }),
  },
  task_auto_producer_idol_card_browser: {
    component: IdolCardBrowser,
    buildProps: ({ section, sectionName, save }) => ({
      data: section,
      taskName: sectionName.replace(/^task__/, ''),
      save,
    }),
  },
}
