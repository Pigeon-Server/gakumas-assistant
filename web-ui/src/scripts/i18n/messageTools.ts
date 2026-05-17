/**
 * 递归合并语言消息对象。
 *
 * @param base 基础语言包
 * @param override 覆盖语言包
 * @returns 合并后的语言包对象
 */
export function deepMergeMessages<T extends Record<string, any>>(base: T, override: Record<string, any>): T {
  const result: Record<string, any> = Array.isArray(base) ? [...base] : { ...base }

  for (const [key, value] of Object.entries(override)) {
    const current = result[key]
    if (isPlainObject(current) && isPlainObject(value)) {
      result[key] = deepMergeMessages(current, value)
      continue
    }
    result[key] = value
  }

  return result as T
}

/**
 * 判断值是否为普通对象。
 *
 * @param value 待判断的值
 * @returns 是否为普通对象
 */
function isPlainObject(value: unknown): value is Record<string, any> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}
