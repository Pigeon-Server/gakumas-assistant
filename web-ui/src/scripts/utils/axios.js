/**
 * 基于axios封装
 * 新增请求拦截 响应拦截
 */
import axios from 'axios'

import message from '@/scripts/utils/message.js'
import { translateKey } from '@/scripts/i18n/translate'

/* 远程地址前缀 */
const axiosplus = axios.create()

/**
 * 响应成功拦截器
 * @param result
 * @return {Promise<never>|*}
 */
const responseSuccess = result => {
  const res = result.data
  let reject = null
  if (res.status === false) {
    message.showApiErrorMsg(res.msg ?? res.message ?? translateKey('backend.api.genericError')).then(() => {})
    reject = Promise.reject(result)
    return reject
  }
  result.message = res.message
  result.data = res?.data
  return result
}

/**
 * 响应失败拦截器
 * @param err
 * @return {Promise<never>}
 */
const responseError = err => {
  if (!err.response) {
    console.debug('[API] request failed before response received', err?.message)
    return Promise.reject(err)
  }
  if (err.response.status === 403) {
    message.showApiErrorMsg(err.response.data.msg ?? err.response.data.message ?? translateKey('backend.api.genericError'))
    setTimeout(() => {
      // window.location.href = '/login'
    }, 1000)
  }
  if (err.response.status === 405) {
    message.showApiErrorMsg(err.response.data.msg ?? err.response.data.message ?? translateKey('backend.api.genericError'))
  }
  if (err.response.status === 500) {
    message.showApiErrorMsg(err.response.data.msg ?? err.response.data.message ?? translateKey('backend.api.genericError'))
  }
  return Promise.reject(err)
}

/**
 * 请求开始拦截器
 * @param config
 * @return {*}
 */
const requestStart = config => {
  return config
}

/**
 * 请求失败拦截器
 * @param err
 * @return {Promise<never>}
 */
const requestError = err => {
  /* 请求失败 */
  return Promise.reject(err)
}

//  响应拦截器
axiosplus.interceptors.response.use(responseSuccess, responseError)
//  请求拦截器
axiosplus.interceptors.request.use(requestStart, requestError)

export default axiosplus
