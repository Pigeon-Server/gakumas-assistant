# 任务配置自定义输入组件

任务配置现在支持“后端元数据驱动渲染 + 前端自定义组件注册”。

目标：

- 大部分任务配置只需要在 `src/entity/Config.py` 里声明 `ConfigItemUI`。
- 少数复杂输入（选卡、白名单、多选物品）通过注册自定义 Vue 组件接入。
- 新任务不需要再手写 `task_list.vue` 的组件映射。

## 渲染链路

1. 后端在 `src/entity/Config.py` 为 `task__*` 配置项声明 `ui` 元数据。
2. 配置接口 `/api/config` / `/api/config/{task}` 返回这些元数据。
3. 前端 `web-ui/src/components/lists/config/task_config_section_form.vue` 自动遍历任务配置项。
4. 每个字段交给 `web-ui/src/components/lists/config/task_config_field.vue` 渲染。
5. 如果 `ui.component` 命中自定义组件注册表，则改走注册组件。

## 后端配置元数据

`ConfigItemUI` 现在支持：

- `label`
- `hint`
- `component`
- `component_props`
- `options`
- `visible_if`
- `readonly`
- `resettable`
- `auto_generate`
- `order`

示例：

```python
target_idol_card_id = ConfigItem(
    default_value="",
    data_type=str,
    ui=ConfigItemUI(
        label="目标偶像卡",
        hint="留空使用默认选中的卡",
        component="task_auto_producer_idol_card_browser",
        order=30,
    ),
)
```

带额外参数的示例：

```python
enhance_ssr_max_level = ConfigItem(
    default_value=60,
    data_type=int,
    ui=ConfigItemUI(
        label="SSR 最大强化等级",
        component="slider",
        component_props={"min": 1, "max": 60, "step": 1, "thumb_label": "always"},
        visible_if={"task__auto_enhancement_support_card.enhance_ssr": True},
        order=11,
    ),
)
```

## 内置组件类型

`task_config_field.vue` 当前支持这些内置 `component`：

- `switch`
- `select`
- `text`
- `number`
- `textarea`
- `slider`
- `time`

如果未显式指定 `component`：

- `bool` 默认走 `switch`
- 有 `options` 默认走 `select`
- 其他默认走 `text`

## 自定义组件注册

注册表文件：

- `web-ui/src/components/lists/config/task_custom_component_registry.ts`

新增一个自定义组件时：

1. 编写 Vue 组件。
2. 在注册表里导入它。
3. 为一个 `ui.component` 名称建立映射。

示例：

```ts
import MyPicker from '@/components/lists/config/task_settings/my_task/my_picker.vue'

export const taskCustomComponentRegistry = {
  task_my_picker: {
    component: MyPicker,
    buildProps: ({ section, sectionName, fieldName, item, save }) => ({
      data: section,
      fieldName,
      label: item?.ui?.label,
      save,
    }),
  },
}
```

然后在后端这样声明：

```python
ui=ConfigItemUI(
    label="我的复杂输入",
    component="task_my_picker",
)
```

## `buildProps` 上下文

注册表的 `buildProps(context)` 会拿到这些字段：

- `config`: 整份配置对象
- `section`: 当前任务配置节，例如 `task__auto_producer`
- `sectionName`: 当前配置节名字
- `fieldName`: 当前字段名
- `item`: 当前字段的配置项对象
- `save`: 可选的保存函数

建议自定义组件尽量只依赖这些入参，不要把任务名、字段路径硬编码在组件内部。

## 自定义组件约定

推荐遵守以下约定：

- 复杂组件直接接收 `data: section`，自行读写 `data.<field>.value`
- 如果组件打开弹窗、选择器、异步面板，修改值后应主动调用 `save?.()`
- 简单输入优先复用内置组件，不要为了一个 `select` 再写自定义组件

`idol_card_browser.vue` 就是一个参考：它在值变化且没有 blur 事件可依赖时，会主动调用保存。

## 什么时候该写自定义组件

适合自定义组件的场景：

- 需要远程搜索 / 异步加载数据
- 多选卡片、白名单、资源浏览器
- 需要图片、描述、富列表项
- 需要弹窗辅助选择

不适合的场景：

- 单纯布尔开关
- 普通文本输入
- 只有几个固定选项的下拉框
- 带 `min/max/step` 的普通数值滑块

## 当前已接入的自定义组件

- `task_auto_purchase_item_selector`
- `task_auto_enhancement_support_card_whitelist`
- `task_auto_producer_idol_card_browser`

## 调试建议

- 如果前端看到“未注册的配置组件”，先检查 `ui.component` 名字是否和注册表一致。
- 如果字段不显示，先检查 `visible_if` 的路径和值是否正确。
- 如果组件改值后没有保存，优先检查是否需要主动调用 `save?.()`。
