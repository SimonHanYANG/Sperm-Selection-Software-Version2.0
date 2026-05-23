# 精子实验数据库查看器 (Sperm Database Viewer)

一个纯前端 HTML 数据库可视化工具，用于查看和编辑由精子优选系统 (SpermSelectionV2) 生成的 SQLite 实验数据库。

## 功能特性

### 数据库加载
- 支持通过文件选择器加载 `.db` 文件
- 支持拖拽 `.db` 文件到页面直接加载
- 使用 sql.js (WebAssembly) 在浏览器端直接解析 SQLite 数据库，无需服务器

### 数据浏览
- **三张数据表**: 精子记录表 (sperm_records)、形态学历史 (morphology_history)、实验元数据 (experiment_meta)
- **双语表头**: 每列显示英文名称和中文注释
- **分页浏览**: 支持 25/50/100/200 条每页
- **列排序**: 点击列头切换升序/降序
- **实时搜索**: 在所有列中进行模糊搜索

### 增删改查 (CRUD)
- **新增记录**: 点击"新增"按钮，弹出表单填写各字段
- **编辑记录**: 双击单元格直接编辑，或点击编辑按钮打开完整表单
- **删除记录**: 点击删除按钮，确认后删除
- **导出数据库**: 将修改后的数据库导出为新的 `.db` 文件

### 数据可视化
- 运动学等级分布 (柱状图)
- 形态学等级分布 (饼图)
- VSL vs ALH 散点图 (按运动学等级着色)
- 复合评分分布 (直方图)
- 头部尺寸散点图 (head_length vs head_width)
- 测量次数 vs 复合评分 (散点图)
- 描述性统计表 (均值、标准差、最小值、中位数、最大值)

### 其他功能
- **SQL 控制台**: 支持自定义 SQL 查询
- **导出 CSV**: 将当前表格数据导出为 CSV 文件
- **统计概览**: 首页显示总追踪数、活跃数、候选池数、平均测量次数

## 技术栈

| 技术 | 用途 | 版本 |
|------|------|------|
| [sql.js](https://github.com/sql-js/sql.js) | 浏览器内 SQLite 引擎 (WebAssembly) | 1.11.0 |
| [Chart.js](https://www.chartjs.org/) | 数据可视化图表 | 4.4.7 |
| [Tailwind CSS](https://tailwindcss.com/) | UI 样式框架 | CDN latest |

所有依赖均通过 CDN 加载，无需安装 Node.js 或任何构建工具。

## 使用方法

### 1. 启动方式

**方式一: 直接打开**
```
双击 SpermDatabaseVisualization/index.html
```
> 注意: 由于 sql.js 需要加载 WASM 文件，某些浏览器在 `file://` 协议下可能无法正常工作。如果遇到问题，请使用方式二。

**方式二: 本地 HTTP 服务器 (推荐)**
```bash
# 使用 Python
cd SpermDatabaseVisualization
python -m http.server 8080

# 或使用 Node.js
npx serve SpermDatabaseVisualization

# 或使用 VS Code 的 Live Server 扩展
```
然后在浏览器中访问 `http://localhost:8080`

### 2. 加载数据库

1. 打开页面后，点击右上角 **"加载数据库"** 按钮
2. 在弹出的文件选择器中，导航到 `SpermDatabase/` 目录
3. 选择一个 `.db` 文件 (推荐选择文件较大的，如 `sperm_experiment_20260522_180713.db`)
4. 也可以直接将 `.db` 文件拖拽到页面中央的虚线框区域

### 3. 浏览数据

- 加载后默认显示 **精子记录表** 标签页
- 点击顶部标签切换不同表格或可视化视图
- 在搜索框中输入关键词实时过滤数据
- 点击列头进行排序
- 使用底部分页控件翻页

### 4. 编辑数据

**编辑单个单元格:**
1. 双击要编辑的单元格
2. 输入新值，按 `Enter` 确认或按 `Escape` 取消

**编辑整行:**
1. 点击行首的编辑图标 (铅笔)
2. 在弹出的模态框中修改字段
3. 点击"确认"保存

**新增记录:**
1. 点击工具栏的"新增"按钮
2. 填写表单字段
3. 点击"确认"添加

**删除记录:**
1. 点击行首的删除图标 (垃圾桶)
2. 在确认对话框中点击"确定"

### 5. 导出数据

- **导出 CSV**: 点击工具栏的"导出CSV"按钮，下载当前表格数据
- **导出数据库**: 点击"导出数据库"按钮，下载包含所有修改的 `.db` 文件

### 6. SQL 控制台

1. 切换到 "SQL 控制台" 标签页
2. 在文本框中输入 SQL 语句
3. 点击"执行查询"按钮
4. 结果将显示在下方表格中

示例查询:
```sql
-- 查看候选池中的高评分精子
SELECT track_id, vsl, alh, composite_score, morphology_grade
FROM sperm_records
WHERE in_candidate_pool = 1
ORDER BY composite_score DESC
LIMIT 20

-- 查看某个精子的形态学变化历史
SELECT frame_number, head_length, head_width, grade
FROM morphology_history
WHERE track_id = 1
ORDER BY frame_number

-- 统计各运动学等级的数量
SELECT kinematic_grade, COUNT(*) as count
FROM sperm_records
GROUP BY kinematic_grade
ORDER BY kinematic_grade
```

## 数据库表结构

### sperm_records (精子记录表)

| 列名 | 类型 | 中文说明 |
|------|------|----------|
| track_id | INTEGER PK | 轨迹ID |
| vsl | REAL | 直线速度 (um/s) |
| alh | REAL | 侧摆幅度 (um) |
| kinematic_grade | INTEGER | 运动学等级 (1-6, 1最优) |
| pos_x | REAL | X坐标 (像素) |
| pos_y | REAL | Y坐标 (像素) |
| head_length | REAL | 头部长度 (um) |
| head_width | REAL | 头部宽度 (um) |
| head_ratio | REAL | 头部长宽比 |
| head_area | REAL | 头部面积 (um^2) |
| neck_width | REAL | 颈部宽度 (um) |
| neck_length | REAL | 颈部长度 (um) |
| neck_head_angle | REAL | 头颈角度 (度) |
| neck_bent_angle | REAL | 颈部弯曲角度 (度) |
| morphology_grade | INTEGER | 形态学等级 (4最优, 5异常, -1未分级) |
| morphology_measurement_count | INTEGER | 测量次数 |
| in_candidate_pool | INTEGER | 是否在候选池 (0/1) |
| scheduling_score | REAL | 调度得分 |
| composite_score | REAL | 复合评分 (0-1) |
| first_seen | REAL | 首次出现时间 (Unix时间戳) |
| last_updated | REAL | 最后更新时间 (Unix时间戳) |
| is_active | INTEGER | 是否活跃 (0/1) |

### morphology_history (形态学历史)

| 列名 | 类型 | 中文说明 |
|------|------|----------|
| id | INTEGER PK AUTO | 记录ID |
| track_id | INTEGER | 轨迹ID |
| timestamp | REAL | 时间戳 |
| frame_number | INTEGER | 帧号 |
| head_length | REAL | 头部长度 (um) |
| head_width | REAL | 头部宽度 (um) |
| head_ratio | REAL | 头部长宽比 |
| head_area | REAL | 头部面积 (um^2) |
| neck_width | REAL | 颈部宽度 (um) |
| neck_length | REAL | 颈部长度 (um) |
| neck_head_angle | REAL | 头颈角度 (度) |
| neck_bent_angle | REAL | 颈部弯曲角度 (度) |
| vsl | REAL | 直线速度 (um/s) |
| alh | REAL | 侧摆幅度 (um) |
| grade | INTEGER | 等级 (4/5/-1) |

### experiment_meta (实验元数据)

| 列名 | 类型 | 中文说明 |
|------|------|----------|
| key | TEXT PK | 键 |
| value | TEXT | 值 |

## 注意事项

1. **WAL 文件**: sql.js 无法读取 WAL 日志文件。如果数据库正在被其他程序写入，部分数据可能不可见。建议在主程序停止后再打开数据库查看。

2. **内存操作**: sql.js 将整个数据库加载到内存中，所有编辑操作都在内存中进行，不会修改原始文件。需要点击"导出数据库"才能保存修改。

3. **浏览器兼容性**: 推荐使用 Chrome、Edge 或 Firefox 最新版本。Safari 对 WASM 的支持可能有差异。

4. **大文件**: 当前数据库文件通常在 1-2MB 左右，加载速度很快。如果遇到特别大的数据库文件 (>50MB)，加载可能需要几秒。

## 文件结构

```
SpermDatabaseVisualization/
    index.html          -- 主应用文件 (HTML + CSS + JavaScript)
    README.md           -- 本文件
```

## 项目关系

本项目是 SpermSelectionV2 精子优选系统的数据库可视化子项目。数据库文件由主系统的 `sperm_registry.py` 模块生成，存储在 `SpermDatabase/` 目录中。

```
SpermSelectionV2_ForDemo_20250815/
    sperm_registry.py           -- 数据库写入模块
    SpermDatabase/              -- 数据库文件目录
        sperm_experiment_*.db   -- 实验数据库文件
    SpermDatabaseVisualization/ -- 本项目
        index.html
        README.md
```
