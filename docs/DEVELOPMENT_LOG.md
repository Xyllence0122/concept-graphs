# Development Log

這份文件依日期記錄 D455 語意建圖專案每次完成的工作、驗證結果、已知問題與下一步。
最新紀錄放在最上方；大型 rosbag、模型權重、地圖資料庫與建置產物只記錄資訊，不提交到 Git。

## 2026-10-22 — D455 語意建圖原型與可重現環境

### 本次目標

- 先在單機完成 Intel RealSense D455 的 RGB-D 資料擷取與語意建圖流程。
- 將偵測物件從相機座標轉換到 RTAB-Map 的 `map` 座標。
- 建立之後可移植到另一台 Thor4／AMR 電腦的 ROS 2 Jazzy 專案與安裝方式。
- 將原始 ConceptGraphs 論文程式和目前 ROS 2 實作分開管理。

### 完成內容

#### 1. D455 RGB-D 資料來源

- 使用 RealSense Viewer 驗證 D455 彩色影像、深度影像及裝置連線。
- 使用 `realsense2_camera` 發布彩色影像、對齊至彩色相機的深度影像與相機內參。
- 實測彩色與 aligned-depth topic 約為 30 Hz。
- 錄製 `d455_static_scene_02` rosbag：長度約 66.09 秒，共 7,925 則訊息。
- rosbag 包含彩色影像、aligned depth、兩組 `CameraInfo` 與 `/tf_static`；大型資料沒有提交到 Git。

#### 2. RTAB-Map 定位與建圖

- 接通 D455 RGB-D topics 至 RTAB-Map RGB-D odometry。
- 成功產生 `/rtabmap/odom`、`map` frame、點雲地圖與 RTAB-Map 資料庫。
- 驗證 `d455_link` 到 `d455_color_optical_frame` 的相機 TF。
- 完成室內環境掃描，確認相機移動時可累積三維地圖。

#### 3. YOLO 與三維物件投影

- 建立 `cg_depth_projection` ROS 2 package，驗證深度 ROI 反投影。
- 建立 `cg_object_segmentation` ROS 2 package，以 YOLO segmentation mask 擷取物件像素。
- 將 mask、aligned depth 與相機內參轉換為相機座標系中的三維 centroid 和 bounding box。
- 將三維偵測結果由 `d455_color_optical_frame` 轉換到 `map` frame。
- 建立 label 與距離為基礎的持續物件追蹤及 JSON graph publisher。
- 在 RViz 顯示點雲與 `MarkerArray` 物件標記。

主要輸出：

| Topic | 用途 |
| --- | --- |
| `/concept_graph/debug/segmentation_image` | YOLO segmentation 除錯畫面 |
| `/concept_graph/objects/detections_3d_json` | 相機座標系的三維偵測 |
| `/concept_graph/objects/detections_map_json` | `map` 座標系的三維偵測 |
| `/concept_graph/graph/json` | 持續累積的語意物件節點 |
| `/concept_graph/graph/markers` | RViz 語意物件標記 |
| `/rtabmap/cloud_map` | RTAB-Map 點雲地圖 |

#### 4. 可重現環境與移植準備

- 固定 ROS、Python、PyTorch 與 AI 套件版本，保存在
  [`../ros2_ws/environment/`](../ros2_ws/environment/)。
- 提供 native bootstrap、環境檢查、build/test 及完整管線啟動腳本。
- 提供 CPU 與 CUDA 12.8 兩種 PyTorch requirements。
- 加入 Dockerfile、Compose 設定與 ROS 2 Jazzy GitHub Actions workflow。
- ROS 2 使用 CycloneDDS 與非預設的 `ROS_DOMAIN_ID=73`。
- 單機測試使用 `LOCALHOST` discovery；移到相機電腦與 AMR 分離的架構時，兩端需改用 `SUBNET`。

已驗證的主要版本：

| 元件 | 版本／設定 |
| --- | --- |
| 平台 | x86_64、Ubuntu 24.04 |
| ROS 2 | Jazzy |
| RealSense | D455、firmware 5.17.3.10 |
| librealsense / realsense-ros | 2.58.4 / 4.58.1 |
| RTAB-Map ROS | 0.23.13 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| Ultralytics | 8.4.154 |
| GPU | RTX 4060 Laptop、driver 595.99.02 |

完整快照位於
[`../ros2_ws/environment/versions-tested.yaml`](../ros2_ws/environment/versions-tested.yaml)。

#### 5. Repository 整理

- 目前開發的 ROS 2 程式保留在 repository 根目錄下的 `ros2_ws/`。
- 原始 ConceptGraphs 論文實作完整移至 `upstream_concept_graphs/`。
- 原始專案檔案逐檔雜湊核對一致，並保留原 README 與 MIT license。
- 分離兩套程式的安裝說明及 `.gitignore`，避免混用舊版 Conda/CUDA 環境與 ROS 2 Python 3.12 環境。

### 驗證結果

- `cg_depth_projection` 與 `cg_object_segmentation` 均可由 `colcon` 建置。
- 共 21 項單元測試通過，0 error、0 failure、0 skipped。
- 完整 launch smoke test 可啟動 D455／RTAB-Map／YOLO／三維投影／object graph nodes。
- YOLO 可載入指定模型並使用 NVIDIA GPU。
- `/concept_graph/objects/detections_map_json` 可輸出 `frame_id: "map"` 的物件。
- `/concept_graph/graph/json` 可持續輸出物件節點；目前 `edges` 為空陣列。

相關提交：

- [`b0b108e`](https://github.com/Xyllence0122/concept-graphs/commit/b0b108ef63a13ca91ad69d482512c835f4162ed8) — 建立可重現的 ROS 2 D455 語意建圖實作。
- [`7714237`](https://github.com/Xyllence0122/concept-graphs/commit/7714237a51fed7dced9a3b510e8af8b7f77f33f4) — 分離原始研究程式與目前 ROS workspace。

### 已知限制

- graph 尚未產生物件間的空間或語意 relation edges。
- label 加距離的 greedy association 可能在遮擋、漂移或 loop closure 後產生重複節點。
- `person` 等動態物件尚未從持續地圖中排除。
- object graph 目前保存在記憶體，尚未匯出成持久化 graph database。
- 尚未依物件位置、機器人 footprint 與 costmap 計算安全的 Nav2 `PoseStamped` goal。

### 下一步

1. 改善物件資料關聯，加入時間穩定性、三維 IoU 與 loop-closure 後的重複節點合併。
2. 建立空間 relation edges，例如 `on`、`inside`、`near`、`left_of` 與 `right_of`。
3. 將語意物件位置投影到 Nav2 costmap，搜尋可通行的觀察位置並輸出 `PoseStamped`。
4. 在第二台 Thor4／AMR 電腦以相同 `ROS_DOMAIN_ID=73`、CycloneDDS 與 `SUBNET` discovery 驗證跨機通訊。

---

## 新紀錄模板

後續開發完成後，複製以下區塊並插入本文件最上方：

```markdown
## YYYY-MM-DD — 本次主題

### 本次目標

- 

### 完成內容

- 

### 驗證結果

- 執行的命令：
- 預期結果：
- 實際結果：

### 問題與決策

- 

### 下一步

1. 
```
