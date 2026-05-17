<template>
  <div ref="containerRef" class="websocket_view">
    <canvas ref="canvasRef" class="websocket_view-canvas" />
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { useTheme } from 'vuetify'
import { wsService } from '@/scripts/utils/websocket.ts'
import { translateKey } from '@/scripts/i18n/translate'

const canvasRef = ref(null)
const containerRef = ref(null)
const vuetifyTheme = useTheme()
let lastBuffer = null
let resizeObserver = null
let renderSeq = 0

/**
 * 判断当前是否处于暗色主题。
 *
 * @returns 是否为暗色主题
 */
function isDarkTheme() {
  return vuetifyTheme.global.name.value === 'dark'
}

function parseBinaryData (buffer) {
  const bytes = new Uint8Array(buffer)
  let comma1 = -1, comma2 = -1

  for (const [i, byte] of bytes.entries()) {
    if (byte === 44) {
      if (comma1 === -1) comma1 = i
      else {
        comma2 = i; break
      }
    }
  }

  if (comma1 === -1 || comma2 === -1) throw new Error(translateKey('websocket.invalidBinaryFormat'))

  const width = Number.parseInt(String.fromCodePoint(...bytes.subarray(0, comma1)), 10)
  const height = Number.parseInt(String.fromCodePoint(...bytes.subarray(comma1 + 1, comma2)), 10)
  const imageBytes = bytes.subarray(comma2 + 1)

  return { width, height, imageBytes }
}

async function renderToCanvas (buffer) {
  const canvas = canvasRef.value
  const container = containerRef.value
  if (!canvas || !container) return

  const currentSeq = ++renderSeq
  const { width, height, imageBytes } = parseBinaryData(buffer)
  const ctx = canvas.getContext('2d')
  const blob = new Blob([imageBytes], { type: 'image/bmp' })
  const bitmap = await createImageBitmap(blob)
  if (currentSeq !== renderSeq) {
    bitmap.close()
    return
  }

  const dpr = window.devicePixelRatio || 1
  const rect = container.getBoundingClientRect()
  const scale = Math.min(rect.width / width, rect.height / height)

  const newWidth = Math.round(width * scale)
  const newHeight = Math.round(height * scale)

  canvas.width = newWidth * dpr
  canvas.height = newHeight * dpr
  canvas.style.width = `${newWidth}px`
  canvas.style.height = `${newHeight}px`

  // 重置 transform 避免 scale 累积
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.scale(dpr, dpr)
  // 下采样时启用高质量平滑，避免调试小字在预览面板里发虚/锯齿明显。
  // 放大或原尺寸时关闭平滑，尽量保留原始像素细节。
  ctx.imageSmoothingEnabled = scale < 1
  ctx.imageSmoothingQuality = scale < 1 ? 'high' : 'low'

  const offsetX = (newWidth - width * scale) / 2
  const offsetY = (newHeight - height * scale) / 2

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(bitmap, offsetX, offsetY, width * scale, height * scale)
  bitmap.close()

  lastBuffer = buffer
}

function drawPlaceholder (text) {
  const canvas = canvasRef.value
  const container = containerRef.value
  if (!canvas || !container) return

  const ctx = canvas.getContext('2d')
  const rect = container.getBoundingClientRect()

  const dpr = window.devicePixelRatio || 1
  canvas.width = rect.width * dpr
  canvas.height = rect.height * dpr
  canvas.style.width = `${rect.width}px`
  canvas.style.height = `${rect.height}px`

  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.scale(dpr, dpr)

  ctx.fillStyle = isDarkTheme() ? '#212121' : '#f6f8fb'
  ctx.fillRect(0, 0, rect.width, rect.height)

  ctx.fillStyle = isDarkTheme() ? '#b0bec5' : '#5f6870'
  ctx.font = '20px sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, rect.width / 2, rect.height / 2)
}

/**
 * 重绘当前预览内容。
 */
function redrawPreview() {
  if (lastBuffer) {
    void renderToCanvas(lastBuffer)
    return
  }
  drawPlaceholder(translateKey('websocket.waitingServer'))
}



onMounted(() => {
  wsService.onBinary(data => {
    renderToCanvas(data)
  })

  const container = containerRef.value
  if (!container) return

  resizeObserver = new ResizeObserver(() => {
    redrawPreview()
  })
  resizeObserver.observe(container)
  redrawPreview()
})

watch(
  () => vuetifyTheme.global.name.value,
  () => {
    redrawPreview()
  },
)

onUnmounted(() => {
  if (resizeObserver) {
    resizeObserver.disconnect()
    resizeObserver = null
  }
})
</script>

<style scoped>
.websocket_view {
  flex: 1 1 auto;
  min-height: 0;
  width: 100%;
  overflow: hidden;
  position: relative;
  background: rgb(var(--v-theme-surface));
  border-radius: 18px;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.06);
  box-shadow: 0 10px 24px rgba(31, 37, 43, 0.05);
}

.websocket_view-canvas {
  display: block;
  margin: 0 auto;
  width: 100%;
  height: 100%;
}
</style>
