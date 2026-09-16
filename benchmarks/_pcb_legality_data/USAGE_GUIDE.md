# PCB Legality Data Package - 使用指南

## 📦 数据包内容

为75个PCB benchmark提取的完整原始数据，用于placement合法性、间距、旋转、NSLW、布线后评估。

### 目录结构

```
benchmarks/_pcb_legality_data/
├── _summary.json                          # 所有板的汇总信息
├── README.md                              # 数据包说明
│
└── <board_name>/                          # 每个板的目录
    ├── <board>_legality_data.json         # 结构化评估数据（主文件）
    └── originals/                         # 原始文件备份
        ├── <board>.nodes                  # Bookshelf nodes
        ├── <board>.nets                   # Bookshelf netlist
        ├── <board>.pl                     # Expert placement
        ├── <board>.wts                    # Net weights (可选)
        ├── <board>_pcb_autoplace.json     # pcb_autoplace原始JSON (如果有)
        └── *.kicad_pcb                    # KiCad原始文件 (如ns-place的bm9)
```

### 生成状态

- ✅ **74/75 boards** 数据已提取
- ⚠️ `carrier_template` 跳过（0 nets，无法评估）

---

## 📋 legality_data.json 字段说明

### 1. Board信息
```json
{
  "name": "board_name",
  "origin": "pcb_autoplace|sa_pcb|nsplace_bm9|unknown",
  
  "board": {
    "bbox_mm": [xmin, ymin, xmax, ymax],     // 板框边界
    "width_mm": 22.098,                       // 板宽度
    "height_mm": 21.463,                      // 板高度
    "grid_mm": 1.0                            // 网格尺寸
  }
}
```

### 2. Design Rules（设计规则）
```json
{
  "design_rules": {
    "num_layers": 2,                          // 层数（默认2层）
    "spacing_rules": {},                      // 间距规则（如果有）
    "courtyard_clearance_mm": 0.25,           // Courtyard间隙（默认0.25mm）
    "min_trace_width_mm": 0.2,                // 最小走线宽度
    "min_clearance_mm": 0.2                   // 最小间距
  }
}
```

**注意**：大部分board使用默认值（0.25mm courtyard clearance）

### 3. Components（元件列表）
```json
{
  "components": [
    {
      "ref": "U1",                            // 元件参考标号
      "internal_id": "o0",                    // 内部ID（用于ISPD格式）
      "size_mm": [3.2, 1.5],                  // 元件尺寸 [width, height]
      "footprint": "Capacitors_SMD:C_0805",   // Footprint名称
      "type": "capacitor",                    // 元件类型
      "terminal": true,                       // 是否为terminal节点
      
      "expert_placement": {                   // Expert placement（如果有）
        "xy_mm": [149.606, 100.863],          // 坐标（绝对mm）
        "rotation": -270.0                    // 旋转角度（度）
      },
      
      "current_placement": {                  // 当前placement（来自.pl文件）
        "x_mm": 149.5,
        "y_mm": 100.8,
        "orientation": "N"                    // N/S/E/W/FN/FS/FE/FW
      },
      
      "pads": [                               // Pad信息（如果有）
        {
          "net": "GND",                       // 网络名
          "rel_mm": [0.95, 0.0],              // 相对元件中心的偏移
          "name": "1"                         // Pad编号
        }
      ],
      
      "allowed_sides": [],                    // 允许的放置面
      "fixed": false                          // 是否固定
    }
  ]
}
```

### 4. Netlist（网表）
```json
{
  "netlist": [
    {
      "name": "GND",                          // 网络名
      "degree": 15,                           // 连接数（扇出）
      "connections": [
        {
          "ref": "U1",                        // 元件参考标号
          "internal_id": "o0",                // 内部ID
          "offset_mm": [0.95, 0.0]            // Pad偏移
        }
      ]
    }
  ]
}
```

### 5. Statistics（统计信息）
```json
{
  "statistics": {
    "num_components": 54,                     // 元件总数
    "num_nets": 30,                           // 网络总数
    "num_pins": 120                           // Pin总数
  }
}
```

---

## 🔧 评估工具

### 1. 单板评估

```bash
cd E:\claude_workspace\converters

python evaluate_legality.py \
  ../benchmarks/_pcb_legality_data/<board>/<board>_legality_data.json \
  ../results/<baseline>/<board>/<board>.pl
```

**输出**：
- 终端打印：边界/重叠/间距违规统计
- `legality_report.json`：详细报告

**示例**：
```bash
python evaluate_legality.py \
  ../benchmarks/_pcb_legality_data/esp8266_wi07_3_adapter_esp/esp8266_wi07_3_adapter_esp_legality_data.json \
  ../results/maskplace/esp8266_wi07_3_adapter_esp/esp8266_wi07_3_adapter_esp.pl
```

### 2. 批量评估

评估一个baseline的所有板：
```bash
python batch_evaluate_legality.py maskplace
```

评估多个baseline：
```bash
python batch_evaluate_legality.py maskplace chipformer sa_pcb
```

**输出**：
- 每个板的`legality_report.json`
- `results/_legality_summary.json`：所有baseline的汇总对比

---

## 📊 评估指标

### 1. Board Boundary Check（板框检查）
- **检查内容**：所有元件是否在板框内
- **违规**：元件超出`bbox_mm`边界
- **数据来源**：
  - 板框：`board.bbox_mm`
  - 元件尺寸：`components[].size_mm`
  - Placement：推理结果.pl文件

### 2. Component Overlap（元件重叠）
- **检查内容**：元件包络框是否相互重叠
- **违规**：两个元件的AABB有交集
- **算法**：轴对齐包围盒（AABB）碰撞检测

### 3. Spacing Check（间距检查）
- **检查内容**：元件边缘间距是否满足最小间隙
- **违规**：元件边缘距离 < `courtyard_clearance_mm`
- **默认值**：0.25mm

### 4. HPWL（Half-Perimeter Wire Length）
- **计算公式**：
  ```
  HPWL = Σ_net [(max_x - min_x) + (max_y - min_y)]
  ```
- **用途**：布线长度估算，值越小越好
- **数据来源**：
  - Netlist：`netlist[]`
  - Pin位置：`components[].pads[].rel_mm` + placement坐标

### 5. Rotation（旋转角度）
- **对比**：推理结果 vs expert placement
- **数据来源**：
  - Expert：`components[].expert_placement.rotation`
  - 推理：.pl文件的orientation字段（需解析）

### 6. Net-Specific Metrics（网络级指标）
- **High fanout nets**：度数 > 10的网络
- **Long nets**：HPWL > threshold的网络
- **Critical nets**：关键信号网络（如时钟、复位）

---

## 🐍 Python API示例

### 读取legality data
```python
import json

# 加载数据
board_data = json.load(open('path/to/board_legality_data.json'))

# 访问板框
bbox = board_data['board']['bbox_mm']
print(f"Board: {bbox[2]-bbox[0]:.1f} x {bbox[3]-bbox[1]:.1f} mm")

# 遍历元件
for comp in board_data['components']:
    print(f"{comp['ref']}: {comp['footprint']} @ {comp.get('expert_placement', {}).get('xy_mm', 'N/A')}")

# 访问网表
for net in board_data['netlist']:
    if net['degree'] > 5:
        print(f"High fanout: {net['name']} (degree={net['degree']})")
```

### 自定义评估规则
```python
from evaluate_legality import read_placement_file, compute_hpwl

# 读取placement
placement = read_placement_file('results/maskplace/board/board.pl')

# 自定义间距检查（例如：大型元件需要更大间隙）
def custom_spacing_check(board_data, placement):
    min_clearance_default = 0.25
    min_clearance_large = 0.5  # 大型元件
    
    for comp in board_data['components']:
        size = comp['size_mm']
        if max(size) > 10.0:  # 大型元件
            # 使用更大的间隙要求
            ...
```

### 批量统计
```python
import json
from pathlib import Path

legality_root = Path('benchmarks/_pcb_legality_data')
boards = []

for board_dir in legality_root.iterdir():
    if not board_dir.is_dir():
        continue
    
    data_file = board_dir / f"{board_dir.name}_legality_data.json"
    if data_file.exists():
        data = json.load(open(data_file))
        boards.append({
            'name': data['name'],
            'components': data['statistics']['num_components'],
            'nets': data['statistics']['num_nets'],
            'size': data['board']['width_mm'] * data['board']['height_mm'],
        })

# 按复杂度排序
boards.sort(key=lambda x: x['components'], reverse=True)
for b in boards[:10]:
    print(f"{b['name']}: {b['components']} components")
```

---

## 🎯 Post-Route评估扩展

当前数据包提供placement级别的评估。如需post-route评估，可扩展：

### 需要的额外数据
1. **Routing结果**：
   - Wire segments（走线段）
   - Via位置
   - 层分配

2. **Design Rules**：
   - 各层最小走线宽度
   - 层间间距
   - Via规格

### 可评估的指标
- ✅ **DRC violations**：走线间距、宽度违规
- ✅ **Via count**：过孔数量
- ✅ **Wire length**：实际布线长度（vs HPWL估算）
- ✅ **Congestion**：拥塞热图
- ✅ **Critical path delay**：关键路径延迟

### 接口建议
```python
def evaluate_routing(legality_data, placement, routing_result):
    """
    routing_result格式：
    {
      "segments": [
        {"net": "GND", "layer": 0, "path": [[x1,y1], [x2,y2]], "width": 0.2}
      ],
      "vias": [
        {"net": "GND", "pos": [x, y], "from_layer": 0, "to_layer": 1}
      ]
    }
    """
    pass
```

---

## ⚠️ 已知限制

### 1. 默认值使用
- **Spacing rules**：大部分board使用默认0.25mm（实际设计可能不同）
- **层数**：默认2层（实际可能是4层、6层等）
- **Trace width**：默认0.2mm

**原因**：原始设计文件（Bookshelf格式）不包含完整的design rules

**建议**：如有原始KiCad项目，从`.kicad_pcb`文件提取真实design rules

### 2. Pad旋转未考虑
- `pads[].rel_mm`是未旋转的偏移
- 计算实际pin坐标时需要应用元件旋转

**修正公式**：
```python
import math
def rotate_point(x, y, angle_deg):
    angle = math.radians(angle_deg)
    return (x * math.cos(angle) - y * math.sin(angle),
            x * math.sin(angle) + y * math.cos(angle))

comp_rot = comp['expert_placement']['rotation']
pad_offset = pad['rel_mm']
rotated_offset = rotate_point(pad_offset[0], pad_offset[1], comp_rot)
pin_x = comp_x + rotated_offset[0]
pin_y = comp_y + rotated_offset[1]
```

### 3. Footprint详细信息缺失
- 大部分板只有元件包络框尺寸，没有footprint的详细轮廓
- Courtyard、Fab layer信息不完整

**解决方案**：如需精确评估，从KiCad footprint库加载完整定义

### 4. 多层板支持有限
- 当前只记录了层数，没有层堆栈信息
- 无法评估层间via的合法性

---

## 📚 相关文件

### 生成工具
- `converters/extract_pcb_legality_data.py` - 数据提取工具
- `converters/evaluate_legality.py` - 单板评估
- `converters/batch_evaluate_legality.py` - 批量评估

### Benchmark格式
- `benchmarks/pcb_bookshelf/` - PCB Bookshelf原始格式
- `benchmarks/ispd_bookshelf/` - ISPD Bookshelf格式
- `benchmarks/_meta/` - ID映射和元数据

### 结果目录
- `results/maskplace/` - maskplace推理结果
- `results/chipformer/` - ChiPFormer推理结果
- `results/_legality_summary.json` - 批量评估汇总

---

## 🚀 快速开始

### 1. 评估单个板
```bash
cd E:\claude_workspace\converters
python evaluate_legality.py \
  ../benchmarks/_pcb_legality_data/simple_example/simple_example_legality_data.json \
  ../results/maskplace/simple_example/simple_example.pl
```

### 2. 批量评估maskplace
```bash
python batch_evaluate_legality.py maskplace
```

### 3. 对比所有baseline
```bash
python batch_evaluate_legality.py maskplace chipformer
cat ../results/_legality_summary.json
```

### 4. 自定义评估
```python
import json
from evaluate_legality import evaluate_placement

# 加载数据
report = evaluate_placement(
    'path/to/legality_data.json',
    'path/to/placement.pl'
)

# 访问结果
print(f"Legal: {report['legality']['is_legal']}")
print(f"HPWL: {report['wire_length']['total_hpwl_mm']} mm")
print(f"Violations: {report['legality']}")
```

---

## 📞 支持

如需扩展评估功能或处理特定设计规则，请参考：
1. `evaluate_legality.py`源码
2. IPC-2221/2222设计规范
3. KiCad DRC规则文档

**数据包生成时间**：2024
**包含boards**：74/75
**原始来源**：pcb_autoplace, SA-PCB, ns-place
