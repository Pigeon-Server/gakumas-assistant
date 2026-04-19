import axiosplus from "@/scripts/utils/axios.js";

function freshGet(url, forceFresh = false) {
  if (!forceFresh) {
    return axiosplus.get(url);
  }
  return axiosplus.get(url, {
    params: {
      _fresh: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    },
    headers: {
      'Cache-Control': 'no-cache',
      Pragma: 'no-cache',
    },
  });
}

/**
 * 开始执行任务队列
 * @return {Promise<AxiosResponse<any, any>>}
 */
function start_task_queue() {
  return axiosplus.get("/api/task/start");
}

/**
 * 终止执行任务队列
 * @return {Promise<AxiosResponse<any, any>>}
 */
function stop_task_queue() {
  return axiosplus.get("/api/task/stop");
}

/**
 * 挂起当前任务
 * @return {Promise<AxiosResponse<any, any>>}
 */
function suspend_task() {
  return axiosplus.get("/api/task/suspend");
}

/**
 * 恢复当前挂起任务
 * @return {Promise<AxiosResponse<any, any>>}
 */
function resume_task() {
  return axiosplus.get("/api/task/resume");
}

/**
 * 执行指定任务
 * @param task_name 任务名
 * @return {Promise<axios.AxiosResponse<any>>}
 */
function run_task(task_name) {
  return axiosplus.get(`/api/task/start/${task_name}`);
}

/**
 * 获取后端状态
 * @return {Promise<AxiosResponse<any, any>>}
 */
function get_status() {
  return axiosplus.get("/api/status");
}

/**
 * 退出应用
 * @return {Promise<AxiosResponse<any, any>>}
 */
function shutdown_app() {
  return axiosplus.post("/api/app/shutdown");
}

/**
 * 获取所有已注册的任务
 * @return {Promise<AxiosResponse<any, any>>}
 */
function get_registered_tasks() {
  return axiosplus.get("/api/task/get_registered_tasks")
}

/**
 * 禁用任务
 * @param task_name 任务名
 * @return {Promise<axios.AxiosResponse<any>>}
 */
function disable_task(task_name) {
  return axiosplus.post(`/api/task/disable/${task_name}`);
}

/**
 * 启用任务
 * @param task_name 任务名
 * @returns {Promise<axios.AxiosResponse<any>>}
 */
function enable_task(task_name) {
  return axiosplus.post(`/api/task/enable/${task_name}`);
}

/**
 * 获取配置
 * @return {Promise<axios.AxiosResponse<any>>}
 */
function get_config(){
  return axiosplus.get("/api/config");
}

/**
 * 获取指定任务配置
 * @param task_name 任务名
 * @returns {Promise<axios.AxiosResponse<any>>}
 */
function get_task_config(task_name) {
  return axiosplus.get(`/api/config/${task_name}`);
}

/**
 * 保存配置
 * @param data 配置数据
 * @return {Promise<axios.AxiosResponse<any>>}
 */
function save_config(data){
  return axiosplus.put(`/api/config`, data);
}

/**
 * 保存指定任务的配置
 * @param task_name 任务名
 * @param data 任务配置数据
 * @returns {Promise<axios.AxiosResponse<any>>}
 */
function save_task_config(task_name, data) {
  return axiosplus.put(`/api/config/${task_name}`, data);
}

/**
 * 获取所有ADB设备列表
 * @param only_usb_device 仅获取USB设备
 */
function get_all_adb_device(only_usb_device=false) {
  if (!only_usb_device) {
    return axiosplus.get("/api/adb/devices");
  }
  return axiosplus.get(`/api/adb/devices/usb`);
}

/**
 * 获取所有物品列表
 */
function get_all_item(forceFresh = false) {
  return freshGet("/api/item/list", forceFresh);
}

/**
 * 获取所有偶像卡列表
 */
function get_all_idol_card(forceFresh = false) {
  return freshGet("/api/idol_card/list", forceFresh);
}

/**
 * 获取所有支援卡列表
 */
function get_all_support_card(forceFresh = false) {
  return freshGet("/api/support_card/list", forceFresh);
}

/**
 * 获取游戏资源下载状态
 */
function get_game_asset_status(forceFresh = false) {
  return freshGet("/api/game_asset/status", forceFresh);
}

/**
 * 触发下载偶像卡缩略图
 */
function download_idol_card_assets() {
  return axiosplus.post("/api/game_asset/download_idol_cards");
}

/**
 * 触发下载支援卡缩略图
 */
function download_support_card_assets() {
  return axiosplus.post("/api/game_asset/download_support_cards");
}

/**
 * 触发下载支援卡全尺寸图片
 */
function download_support_card_full_assets() {
  return axiosplus.post("/api/game_asset/download_support_cards_full");
}

/**
 * 按需下载单张支援卡全尺寸图片
 */
function download_single_card_full(cardId) {
  return axiosplus.post(`/api/game_asset/download_card_full/${cardId}`);
}

/**
 * 按需下载单张偶像卡全尺寸图片
 */
function download_single_idol_card_full(cardId, skin = 0) {
  return axiosplus.post(`/api/game_asset/download_idol_card_full/${cardId}?skin=${skin}`);
}

/**
 * 批量下载所有支援卡相关图片（在设置页面触发）
 */
function auto_download_assets() {
  return axiosplus.post("/api/game_asset/auto_download");
}

/**
 * 刷新DMM Player启动参数
 */
function refresh_ddm_player_token() {
  return axiosplus.get("/api/config/tools/refresh_ddm_token");
}

/**
 * 获取资源仓库更新状态
 */
function get_resource_update_status() {
  return axiosplus.get("/api/resource_update/status");
}

/**
 * 手动检查资源仓库更新
 */
function check_resource_updates() {
  return axiosplus.post("/api/resource_update/check");
}

/**
 * 应用资源仓库更新
 */
function apply_resource_updates() {
  return axiosplus.post("/api/resource_update/apply");
}

/**
 * 重置所有配置项
 */

function reset_config() {
  return axiosplus.get("/api/config/tools/reset_config");
}

export default {
  start_task_queue,
  stop_task_queue,
  suspend_task,
  resume_task,
  run_task,
  get_status,
  shutdown_app,
  get_registered_tasks,
  disable_task,
  enable_task,
  get_config,
  save_config,
  get_task_config,
  save_task_config,
  get_all_adb_device,
  get_all_item,
  get_all_idol_card,
  get_all_support_card,
  get_game_asset_status,
  download_idol_card_assets,
  download_support_card_assets,
  download_support_card_full_assets,
  download_single_card_full,
  download_single_idol_card_full,
  auto_download_assets,
  refresh_ddm_player_token,
  get_resource_update_status,
  check_resource_updates,
  apply_resource_updates,
  reset_config,
}
