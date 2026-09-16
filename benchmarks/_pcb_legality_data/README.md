# PCB Legality Evaluation Data

## 数据内容

为每个benchmark提取的完整PCB信息，用于合法性、间距、旋转、NSLW、布线后评估。

**总计**: 74 个板

## 数据结构

```
_pcb_legality_data/
├── _summary.json                          # 所有板的汇总信息
├── README.md                              # 本文件
└── <board_name>/
    ├── <board>_legality_data.json         # 提取的结构化数据
    └── originals/                         # 原始文件副本
        ├── <board>.nodes                  # Bookshelf nodes
        ├── <board>.nets                   # Bookshelf nets
        ├── <board>.pl                     # Expert placement
        ├── <board>.wts                    # Net weights (可选)
        ├── <board>_pcb_autoplace.json     # pcb_autoplace原始JSON (如果有)
        └── *.kicad_pcb                    # KiCad原始文件 (如果有)
```

## legality_data.json 格式

每个板的`<board>_legality_data.json`包含：

```json
{
  "name": "board_name",
  "origin": "pcb_autoplace|sa_pcb|nsplace_bm9",

  "board": {
    "bbox_mm": [xmin, ymin, xmax, ymax],
    "width_mm": 22.098,
    "height_mm": 21.463,
    "grid_mm": 1.0
  },

  "design_rules": {
    "num_layers": 2,
    "spacing_rules": {},
    "courtyard_clearance_mm": 0.25,
    "min_trace_width_mm": 0.2,
    "min_clearance_mm": 0.2
  },

  "components": [
    {
      "ref": "U1",
      "internal_id": "o0",
      "size_mm": [3.2, 1.5],
      "footprint": "Capacitors_SMD:C_0805",
      "type": "capacitor",
      "terminal": true,
      "expert_placement": {
        "xy_mm": [149.606, 100.863],
        "rotation": -270.0
      },
      "pads": [
        {
          "net": "GND",
          "rel_mm": [0.95, 0.0],
          "name": "1"
        }
      ]
    }
  ],

  "netlist": [
    {
      "name": "GND",
      "degree": 15,
      "connections": [
        {
          "ref": "U1",
          "internal_id": "o0",
          "offset_mm": [0.95, 0.0]
        }
      ]
    }
  ],

  "statistics": {
    "num_components": 54,
    "num_nets": 30,
    "num_pins": 120
  }
}
```

## 评估工具可用数据

### 1. 板框检查（Board Outline Legality）
- `board.bbox_mm` - 板边界框
- `components[].size_mm` - 元件尺寸
- `components[].expert_placement.xy_mm` - 放置坐标

**检查**: 所有元件是否在板框内

### 2. 间距规则（Spacing Rules）
- `design_rules.courtyard_clearance_mm` - 默认courtyard间隙
- `design_rules.min_clearance_mm` - 最小间隙
- `components[].size_mm` - 元件包络框

**检查**: 元件之间是否满足最小间距

### 3. 旋转角度（Rotation）
- `components[].expert_placement.rotation` - expert的旋转角度
- 可与推理结果对比

### 4. Pad位置（Pin/Pad Legality）
- `components[].pads[].rel_mm` - pad相对元件中心的偏移
- `components[].pads[].net` - pad的网络连接

**检查**: pad位置经旋转后是否合法

### 5. 网络连接（NSLW - Net-Specific Layout Width）
- `netlist[].connections` - 每个net的所有连接
- `netlist[].degree` - net的度数

**检查**: 高扇出net的布局质量

### 6. 层数（Layer Count）
- `design_rules.num_layers` - 设计层数（默认2层）

## 使用示例

### Python读取
```python
import json

# 读取一个板的数据
board_data = json.load(open('_pcb_legality_data/esp8266_wi07_3_adapter_esp/esp8266_wi07_3_adapter_esp_legality_data.json'))

# 检查板框
bbox = board_data['board']['bbox_mm']
for comp in board_data['components']:
    xy = comp['expert_placement']['xy_mm']
    size = comp['size_mm']
    # 检查是否超出板框
    if xy[0] - size[0]/2 < bbox[0] or xy[0] + size[0]/2 > bbox[2]:
        print(f"Component {comp['ref']} out of board bounds")

# 检查间距
min_spacing = board_data['design_rules']['courtyard_clearance_mm']
components = board_data['components']
for i, c1 in enumerate(components):
    for c2 in components[i+1:]:
        # 计算两个元件的距离
        ...
```

## 注意事项

1. **默认值**: 部分board没有明确的spacing rules，使用了合理默认值（0.25mm courtyard clearance）
2. **层数**: 大部分board默认2层，实际可能不同
3. **坐标系**: 所有坐标都是绝对mm单位，原点在板框左下角
4. **旋转**: 旋转角度单位是度（degrees），逆时针为正
5. **Pad偏移**: pad的`rel_mm`是相对元件中心的偏移，未考虑旋转

## 生成时间

Generated: 1__analog_esr_meter_esr_meter_rev_a
Total boards: 74
