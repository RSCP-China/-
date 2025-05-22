# Production Scheduling System | 生产调度系统

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

A finite capacity production scheduling system with support for operation batching and multi-criteria optimization.

### Core Features

1. Multi-Criteria Optimization
   - Minimize total makespan
   - Prioritize due dates
   - Maximize resource utilization
   - Minimize setup times
   - Configurable weights for each criterion

2. Intelligent Batching
   - Automatic batching of similar operations
   - Setup time optimization
   - Configurable batch size limits
   - Time window-based batching

3. Priority-Based Scheduling
   - Job priority consideration
   - Due date optimization
   - Operation sequence dependencies
   - Resource capacity constraints

### Scheduling Logic

#### Priority System
- Operations are scheduled based on:
  1. Job Priority (lower number = higher priority)
  2. Due Date (earlier = higher priority)
  3. Operation Sequence (maintains manufacturing order)

#### Batching Algorithm
1. Operations are grouped by Part Number and WorkCenter
2. Within each group:
   - Operations are sorted by priority and due date
   - Batches are formed up to max_batch_hours limit
   - Setup time is shared within a batch
   - Batch priority is determined by highest priority operation

#### Resource Management
- Each WorkCenter can have multiple machines
- Machine availability is tracked
- Shift hours and capacity constraints are respected
- Resource utilization is optimized and visualized

### Required Input Files

#### Production Orders (CSV)
- Job Number
- Part Number
- Due Date
- Operation Sequence
- WorkCenter
- Place
- Run Time
- Setup Time
- JobPriority

#### Resources (CSV)
- WorkCenter
- Place
- Available Quantity
- Shift Hours

### Output and Visualization
1. Schedule Results
   - Detailed operation schedule
   - Start and finish times
   - Machine assignments
   - Downloadable CSV format

2. Visual Analytics
   - Interactive Gantt chart
   - Work center utilization heatmap
   - Job tracking and highlighting
   - Performance metrics

---

<a name="chinese"></a>
## 中文

一个具有操作批处理和多准则优化功能的有限产能生产调度系统。

### 核心功能

1. 多准则优化
   - 最小化总完工时间
   - 优先考虑交付日期
   - 最大化资源利用率
   - 最小化设置时间
   - 各优化准则权重可配置

2. 智能批处理
   - 相似操作自动批处理
   - 设置时间优化
   - 可配置批次规模限制
   - 基于时间窗口的批处理

3. 基于优先级的调度
   - 考虑作业优先级
   - 交付日期优化
   - 操作顺序依赖关系
   - 资源产能约束

### 调度逻辑

#### 优先级系统
- 操作调度基于：
  1. 作业优先级（数字越小优先级越高）
  2. 交付日期（越早优先级越高）
  3. 操作顺序（保持制造顺序）

#### 批处理算法
1. 按产品编号和工作中心对操作进行分组
2. 在每个组内：
   - 按优先级和交付日期排序
   - 在最大批次工时限制内形成批次
   - 批次内共享设置时间
   - 批次优先级由最高优先级操作决定

#### 资源管理
- 每个工作中心可以有多台机器
- 跟踪机器可用性
- 遵守班次时间和产能约束
- 优化并可视化资源利用率

### 所需输入文件

#### 生产订单 (CSV)
- 作业编号
- 产品编号
- 交付日期
- 操作顺序
- 工作中心
- 工位
- 运行时间
- 设置时间
- 作业优先级

#### 资源数据 (CSV)
- 工作中心
- 工位
- 可用数量
- 班次时间

### 输出和可视化
1. 调度结果
   - 详细操作调度
   - 开始和结束时间
   - 机器分配
   - 可下载的CSV格式

2. 可视化分析
   - 交互式甘特图
   - 工作中心利用率热图
   - 作业跟踪和突出显示
   - 性能指标