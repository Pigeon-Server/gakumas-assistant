/// <reference types="vite/client" />

import 'axios'

declare module 'axios' {
  interface AxiosResponse<T = any, D = any, H = {}> {
    message?: string
  }
}

interface ImportMeta {
  glob: (
    pattern: string,
    options?: {
      eager?: boolean
      import?: string
      as?: string
    }
  ) => Record<string, unknown>
}
