import gymnasium as gym
import numpy as np
import networkx as nx
from collections import defaultdict, deque

class PCBPlacementEnv(gym.Env):

    def __init__(self,
                 grid_size=128,
                 num_rotations=4,
                 max_pads=16,
                 num_component_types=5,
                 hpwl_range=2000,
                 boundary_mask=None,
                 component_list=None,
                 netlist=None,
                 spacing_rules=None,
                 body_spacing_rules=None,
                 pad_spacing_rules=None,
                 boundary_polygon=None):

        super().__init__()

        # 基本参数
        self.N = grid_size
        self.grid_size = grid_size
        self.num_rotations = num_rotations
        self.max_pads = max_pads
        self.num_component_types = num_component_types
        self.hpwl_range = hpwl_range

        # PCB 数据
        self.boundary_mask = boundary_mask
        self.component_list = component_list or []
        self.netlist = netlist or {}
        self.num_components = len(self.component_list)

        # 间距规则表：{(type1, type2): min_distance} 或 {(comp_id1, comp_id2): min_distance}
        self.spacing_rules = spacing_rules or {}
        # 本体间距规则：如果用户未传入，则自动生成默认规则（所有类型对之间本体间距为2）
        self.body_spacing_rules = body_spacing_rules or {}
        if not self.body_spacing_rules:
            for t1 in range(self.num_component_types):
                for t2 in range(self.num_component_types):
                    self.body_spacing_rules[(t1, t2)] = 2
        # 引脚间距规则：如果用户未传入，则自动生成默认规则（所有类型对之间引脚间距为1）
        self.pad_spacing_rules = pad_spacing_rules or {}
        if not self.pad_spacing_rules:
            for t1 in range(self.num_component_types):
                for t2 in range(self.num_component_types):
                    self.pad_spacing_rules[(t1, t2)] = 1

        # 边界多边形：[(x1,y1), (x2,y2), ...] 连续坐标
        self.boundary_polygon = boundary_polygon

        # Runtime 状态
        self.current_index = 0
        self.current_component = None
        self.illegal = False
        self.hpwl = 0
        self.prev_hpwl = 0
        self.slw = 0
        self.prev_slw = 0
        self.nslw = 0
        self.prev_nslw = 0
        self.current_component_position = None  # render 使用
        self.placed_components = []  # 记录已放置组件的位置 (x, y, w, h, type_id, comp_id)

        # observation space
        self.observation_space = gym.spaces.Dict({
            "view_mask": gym.spaces.Box(
                low=0, high=1, shape=(self.N, self.N), dtype=np.int8
            ),
            "position_mask": gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.num_rotations, self.N, self.N),
                dtype=np.float32
            ),
            "wire_mask": gym.spaces.Box(
                low=-self.hpwl_range, high=self.hpwl_range,
                shape=(self.num_rotations, self.N, self.N),
                dtype=np.float32
            ),
            "component": gym.spaces.Dict({
                "size": gym.spaces.Box(
                    low=0, high=self.N,
                    shape=(2,), dtype=np.int32
                ),
                "pad": gym.spaces.Box(
                    low=0, high=self.N,
                    shape=(self.max_pads, 2), dtype=np.int32
                ),
                "type": gym.spaces.Discrete(self.num_component_types)
            })
        })

        # action: (x, y, rot)
        self.action_space = gym.spaces.MultiDiscrete([self.N, self.N, self.num_rotations])

        # 内部状态矩阵
        self.view_mask = np.zeros((self.N, self.N), dtype=np.int8)
        self.position_mask = np.zeros((self.num_rotations, self.N, self.N), dtype=np.float32)
        self.wire_mask = np.zeros((self.num_rotations, self.N, self.N), dtype=np.float32)
        self.pad_mask = np.zeros((self.N, self.N), dtype=np.int8)  # 标记pads位置




    def _is_point_in_polygon(self, point, polygon):
        """
        使用射线法检查点是否在多边形内。
        point: (x, y)
        polygon: [(x1,y1), (x2,y2), ...]
        """
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


    def _rect_distance(self, bbox1, bbox2):
        """
        计算两个包围盒之间的最小距离。
        bbox: (x1, y1, x2, y2) 其中 (x1,y1) 是左上角，(x2,y2) 是右下角
        注意：在PCB设计中，包围盒接触也算重叠，返回0
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # 计算x方向的距离
        if x1_max < x2_min:
            dx = x2_min - x1_max
        elif x2_max < x1_min:
            dx = x1_min - x2_max
        else:
            return 0.0  # x方向重叠或接触（包括边界接触）
        
        # 计算y方向的距离
        if y1_max < y2_min:
            dy = y2_min - y1_max
        elif y2_max < y1_min:
            dy = y1_min - y2_max
        else:
            return 0.0  # y方向重叠或接触（包括边界接触）
        
        # 返回欧氏距离
        return np.sqrt(dx**2 + dy**2)

    @staticmethod
    def _orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else 2

    @staticmethod
    def _on_segment(p, q, r):
        return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
                min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)

    @classmethod
    def _segments_intersect(cls, p1, q1, p2, q2):
        o1 = cls._orientation(p1, q1, p2)
        o2 = cls._orientation(p1, q1, q2)
        o3 = cls._orientation(p2, q2, p1)
        o4 = cls._orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and cls._on_segment(p1, p2, q1):
            return True
        if o2 == 0 and cls._on_segment(p1, q2, q1):
            return True
        if o3 == 0 and cls._on_segment(p2, p1, q2):
            return True
        if o4 == 0 and cls._on_segment(p2, q1, q2):
            return True
        return False

    @classmethod
    def _segment_intersects_rect(cls, p1, p2, rect):
        x1, y1, x2, y2 = rect
        # If either endpoint is inside rectangle, count as intersection
        if (x1 - 1e-9 <= p1[0] <= x2 + 1e-9 and y1 - 1e-9 <= p1[1] <= y2 + 1e-9):
            return True
        if (x1 - 1e-9 <= p2[0] <= x2 + 1e-9 and y1 - 1e-9 <= p2[1] <= y2 + 1e-9):
            return True

        edges = [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]

        for e1, e2 in edges:
            if cls._segments_intersect(p1, p2, e1, e2):
                return True
        return False

    def _segment_clear(self, p1, p2, ignore_components):
        for px, py, pw, ph, type_id, comp_id in self.placed_components:
            if comp_id in ignore_components:
                continue
            rect = (px, py, px + pw, py + ph)
            if self._segment_intersects_rect(p1, p2, rect):
                return False
        return True

    def _collect_pad_fanouts(self):
        pad_directions = {}
        pad_positions = {}

        for comp in self.components:
            if "placed_pads" not in comp or not comp["placed_pads"]:
                continue
            comp_id = comp["comp_id"]
            for pad_idx, (px, py) in enumerate(comp["placed_pads"]):
                dist_top = py
                dist_bottom = self.N - py
                dist_left = px
                dist_right = self.N - px
                distances = [dist_top, dist_bottom, dist_left, dist_right]
                direction = int(np.argmin(distances))
                pad_directions[(comp_id, pad_idx)] = direction
                pad_positions[(comp_id, pad_idx)] = (float(px), float(py))

        return pad_directions, pad_positions


    def _rotate_pad(self, pad, rot, comp_w, comp_h):
        """
        根据旋转角度调整pad位置。
        rot: 0,1,2,3 对应 0°,90°,180°,270°
        """
        px, py = pad
        if rot == 0:
            return px, py
        elif rot == 1:  # 90° 顺时针
            return py, comp_w - px
        elif rot == 2:  # 180°
            return comp_w - px, comp_h - py
        elif rot == 3:  # 270° 顺时针
            return comp_h - py, px
        else:
            return px, py


    def _get_rotated_dims_and_pads(self, comp, rot):
        """
        根据旋转获取组件的尺寸和pad坐标。
        """
        w, h = comp["size"]
        if rot in [1, 3]:  # 90° or 270°
            w, h = h, w
        pads = [self._rotate_pad(p, rot, comp["size"][0], comp["size"][1]) for p in comp["pad_list"]]
        return w, h, pads


    def reset(self):
        """
        重置环境到纯净的初始状态，保证零历史。
        所有与已放置元件相关的运行时状态都必须重置。
        """

        # ================ 运行时状态重置（开头）================
        # 序列指针重置
        self.current_index = 0
        self.current_component = None
        self.current_component_position = None

        # 已放置元件状态重置
        self.placed_components = []  # 已放置组件列表

        # 根据 placed_components 重建 masks（此时为空，自然得到全零 mask）
        self._rebuild_masks_from_placed_components()

        # 指标重置
        self.hpwl = 0
        self.prev_hpwl = 0
        self.slw = 0
        self.prev_slw = 0
        self.nslw = 0
        self.prev_nslw = 0

        # 状态标志重置
        self.illegal = False

        # ================ 任务加载与初始化 ================
        # 加载任务
        self.load_task()

        # 生成 placement seq
        self.placement_sequence = self.compute_placement_sequence(self.component_list)

        # 设置第一个组件
        self.current_component = self.placement_sequence[0]

        # 构建观测mask
        self.position_mask = self.build_position_mask(self.current_component)
        self.wire_mask = self.build_wire_mask(self.current_component)

        obs = {
            "view_mask": self.view_mask.copy(),
            "position_mask": self.position_mask.copy(),
            "wire_mask": self.wire_mask.copy(),
            "component": {
                "size": np.array(self.current_component["size"], dtype=np.int32),
                "pad": np.array(self.current_component["pad_list"], dtype=np.int32),
                "type": int(self.current_component["type_id"])
            }
        }
        return obs


    def step(self, action):
        x, y, rot = action
        comp = self.current_component

        # 获取旋转后的尺寸
        w, h, _ = self._get_rotated_dims_and_pads(comp, rot)

        # --------------- 检查合法性 ------------------
        if not self._check_boundary(x, y, w, h):
            return self._finish_illegal("boundary_violation")

        if not self._check_overlap(x, y, w, h, rot):
            return self._finish_illegal("overlap_violation")

        if not self._check_spacing(x, y, w, h, rot):
            return self._finish_illegal("spacing_violation")

        # --------------- 放置组件 ----------------------
        self._place_component(x, y, w, h, rot, comp["comp_id"])
        self.current_component_position = (x, y, rot)

        # 根据 placed_components 重建 masks，确保与单一事实来源同步
        self._rebuild_masks_from_placed_components()

        # HPWL 计算
        self.prev_hpwl = self.hpwl
        self.hpwl = self._compute_hpwl()
        # SLW 计算
        self.prev_slw = self.slw
        self.slw = self._compute_slw()
        # NSLW 计算
        self.prev_nslw = self.nslw
        self.nslw = self._compute_nslw()
        reward = (self.prev_hpwl - self.hpwl) + (self.prev_slw - self.slw)

        # 是否结束
        self.current_index += 1
        info = {
            "hpwl": float(self.hpwl),
            "slw": float(self.slw),
            "nslw": float(self.nslw),
        }

        if self.current_index >= self.num_components:
            return self._construct_observation(True), reward, True, info

        # 下一个组件
        self.current_component = self.placement_sequence[self.current_index]

        # 更新 masks
        self.position_mask = self.build_position_mask(self.current_component)
        self.wire_mask = self.build_wire_mask(self.current_component)

        return self._construct_observation(False), reward, False, info


    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)  # 固定dpi
        ax.set_xlim(0, self.N)
        ax.set_ylim(0, self.N)
        ax.set_aspect('equal')
        ax.set_title(f"PCB Placement Environment\nHPWL: {self.hpwl:.1f}, SLW: {self.slw:.1f}")

        # 绘制电路板边界（最外圈）
        boundary_rect = patches.Rectangle((0, 0), self.N, self.N, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(boundary_rect)

        # 绘制多边形边界（如果有）
        if self.boundary_polygon is not None:
            poly = patches.Polygon(self.boundary_polygon, closed=True, fill=False, edgecolor='red', linewidth=1, linestyle='--')
            ax.add_patch(poly)

        # 绘制非法区域（如果有）
        if self.boundary_mask is not None:
            illegal_y, illegal_x = np.where(self.boundary_mask == 0)
            ax.scatter(illegal_x, illegal_y, c='gray', s=1, alpha=0.3, label='Illegal Area')

        # 绘制nets连接线
        net_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for net_idx, net in enumerate(self.G):
            color = net_colors[net_idx % len(net_colors)]
            pads = []
            for comp_id, pad_idx in net:
                comp = self.components[comp_id]
                if pad_idx < len(comp["placed_pads"]):
                    px, py = comp["placed_pads"][pad_idx]
                    pads.append((px, py))
            
            # 绘制连接线
            if len(pads) > 1:
                for i in range(len(pads)):
                    for j in range(i+1, len(pads)):
                        x1, y1 = pads[i]
                        x2, y2 = pads[j]
                        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.6)

        # 绘制所有pads
        for comp in self.components:
            if "placed_pads" in comp and comp["placed_pads"]:
                for px, py in comp["placed_pads"]:
                    ax.scatter(px, py, c='red', s=20, marker='o', edgecolors='black', linewidth=0.5, zorder=5)

        # 绘制已放置组件 - 放在最后，确保在最上层
        for px, py, pw, ph, type_id, comp_id in self.placed_components:
            # 根据组件类型选择边框颜色
            colors = ['red', 'blue', 'green', 'orange', 'purple']  # 边框颜色
            color = colors[type_id % len(colors)]
            rect = patches.Rectangle((px, py), pw, ph, linewidth=3, edgecolor=color, facecolor='none', zorder=10)  # 只显示边框，不填充
            ax.add_patch(rect)
            
            # 显示组件ID
            ax.text(px + pw/2, py + ph/2, f'C{comp_id}', ha='center', va='center', fontsize=12, fontweight='bold', color=color, zorder=11)

        # 绘制当前组件（如果有且未完成）
        if self.current_component_position is not None and self.current_index < self.num_components:
            x, y, r = self.current_component_position
            w, h, rotated_pads = self._get_rotated_dims_and_pads(self.current_component, r)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            
            # 绘制当前组件的pads
            for px, py in rotated_pads:
                ax.scatter(x + px, y + py, c='orange', s=25, marker='s', edgecolors='black', linewidth=1, zorder=6)

        # 添加图例
        legend_elements = [
            patches.Patch(edgecolor='red', facecolor='none', linewidth=2, label='Placed Components'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Pads'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=8, label='Current Component Pads'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='Nets'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()

        if mode == "rgb_array":
            # 使用更简单的方法：保存到PNG然后读取
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            from PIL import Image
            img = Image.open(buf)
            # 确保图像是RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            canvas = np.array(img)
            
            plt.close(fig)
            return canvas
        elif mode == "human":
            plt.savefig('matplotlib_render.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.close(fig)


    def close(self):
        pass

    def load_task(self):
        if self.component_list is None:
            raise ValueError("component_list must be provided")

        if self.netlist is None:
            raise ValueError("netlist must be provided")

        if self.boundary_mask is None and self.boundary_polygon is None:
            # 若 boundary 未提供，默认全合法
            self.boundary_mask = np.ones((self.N, self.N), dtype=np.int8)
        elif self.boundary_mask is None and self.boundary_polygon is not None:
            # 从polygon生成mask
            self.boundary_mask = self._polygon_to_mask(self.boundary_polygon, self.N)

        self.components = self.component_list
        self.G = self.netlist

        # 初始化 placed_pads
        for comp in self.components:
            comp["placed_pads"] = []


    def _polygon_to_mask(self, polygon, size):
        """
        将多边形转换为二进制mask。
        使用保守策略：如果网格中心在polygon内，则为1。
        """
        mask = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            for j in range(size):
                center_x = j + 0.5
                center_y = i + 0.5
                if self._is_point_in_polygon((center_x, center_y), polygon):
                    mask[i, j] = 1
        return mask

    def compute_placement_sequence(self, components):
        """Placement order following the PCBAgent paper heuristics.

        Steps:
        1. Identify chips (type_id == 0) and place them first.
        2. For each chip, order pads clockwise starting from the pad whose
           connected components have the highest weight.
        3. Assign weights to non-chip components via a linear combination of
           normalised size, degree, chip pad round (outer rounds preferred),
           pad radius, and graph level (closer to chip preferred).
        4. Recursively traverse neighbours so that components connected to the
           same chip pad (and their downstream neighbours) are scheduled before
           moving to the next pad.
        5. Fallback to a deterministic rule when no chips are present or when
           components remain disconnected from all chips.
        """

        if not components:
            return []

        id_to_component = {comp["comp_id"]: comp for comp in components}
        comp_ids = set(id_to_component.keys())

        chips = [comp for comp in components if comp.get("type_id") == 0]
        non_chips = [comp for comp in components if comp.get("type_id") != 0]

        # Fallback: no chips detected, retain deterministic ordering based on
        # pad count, area, and component id.
        if not chips:
            def fallback_key(comp):
                pad_count = len(comp.get("pad_list", ()))
                w, h = comp.get("size", (0, 0))
                area = w * h
                return (-pad_count, -area, comp["comp_id"])

            return sorted(components, key=fallback_key)

        # ------------------------------------------------------------------
        # Build adjacency and helper metadata from the netlist.
        # ------------------------------------------------------------------
        adjacency = {comp_id: set() for comp_id in comp_ids}
        component_degree = {comp_id: 0 for comp_id in comp_ids}

        for net in self.G:
            for i in range(len(net)):
                comp_i, _ = net[i]
                if comp_i not in comp_ids:
                    continue
                component_degree[comp_i] += 1
                for j in range(i + 1, len(net)):
                    comp_j, _ = net[j]
                    if comp_j not in comp_ids:
                        continue
                    adjacency[comp_i].add(comp_j)
                    adjacency[comp_j].add(comp_i)

        # Breadth-first levels from all chips.
        INF_LEVEL = float("inf")
        levels = {comp_id: INF_LEVEL for comp_id in comp_ids}
        queue = deque()

        for chip in chips:
            cid = chip["comp_id"]
            levels[cid] = 0
            queue.append(cid)

        while queue:
            current = queue.popleft()
            base_level = levels[current]
            for neighbour in adjacency[current]:
                if levels[neighbour] > base_level + 1:
                    levels[neighbour] = base_level + 1
                    queue.append(neighbour)

        max_level = max((lvl for lvl in levels.values() if np.isfinite(lvl)), default=0)

        # Component areas for normalisation.
        areas = {comp_id: id_to_component[comp_id]["size"][0] * id_to_component[comp_id]["size"][1]
                 for comp_id in comp_ids}
        max_area = max(areas.values()) if areas else 1.0
        max_degree = max(component_degree.values()) if component_degree else 1.0

        # Collect chip pad metadata and connections.
        chip_pad_metadata = {}
        chip_pad_connections = defaultdict(lambda: defaultdict(set))  # chip_id -> pad_idx -> set(comp_id)
        component_chip_links = defaultdict(list)  # comp_id -> [(chip_id, pad_idx)]

        def build_pad_metadata(chip_component):
            w, h = chip_component["size"]
            center_x = w / 2.0
            center_y = h / 2.0
            pads = chip_component.get("pad_list", [])
            metadata = []
            radii = []

            for pad_idx, (px, py) in enumerate(pads):
                dx = (px + 0.5) - center_x
                dy = (py + 0.5) - center_y
                radius = float(np.hypot(dx, dy))
                angle = float(np.arctan2(dy, dx))
                if angle < 0:
                    angle += 2 * np.pi
                metadata.append({
                    "index": pad_idx,
                    "angle": angle,
                    "radius": radius,
                })
                radii.append(radius)

            # Assign round indices by clustering radii.
            if metadata:
                sorted_pairs = sorted(((entry["radius"], idx) for idx, entry in enumerate(metadata)))
                current_round = 0
                previous_radius = None
                for radius, meta_idx in sorted_pairs:
                    if previous_radius is None or abs(radius - previous_radius) > 1.0:
                        current_round += 1
                        previous_radius = radius
                    metadata[meta_idx]["round"] = current_round
            else:
                for entry in metadata:
                    entry["round"] = 1

            return metadata

        for chip in chips:
            chip_id = chip["comp_id"]
            metadata = build_pad_metadata(chip)
            chip_pad_metadata[chip_id] = metadata

        chip_id_set = {chip["comp_id"] for chip in chips}

        # Map nets to pad connections.
        for net in self.G:
            chip_entries = [entry for entry in net if entry[0] in chip_id_set]
            if not chip_entries:
                continue
            other_entries = [entry for entry in net if entry[0] not in chip_id_set]
            if not other_entries:
                continue
            for chip_id, pad_idx in chip_entries:
                if chip_id not in chip_pad_metadata:
                    continue
                for comp_id, _ in other_entries:
                    if comp_id not in comp_ids:
                        continue
                    chip_pad_connections[chip_id][pad_idx].add(comp_id)
                    component_chip_links[comp_id].append((chip_id, pad_idx))

        # Prepare normalisation constants for rounds and radii.
        all_rounds = []
        all_radii = []
        for metadata in chip_pad_metadata.values():
            for entry in metadata:
                all_rounds.append(entry.get("round", 1))
                all_radii.append(entry.get("radius", 0.0))

        max_round = max(all_rounds) if all_rounds else 1
        max_radius = max(all_radii) if all_radii else 1.0

        # Component weights for ordering (non-chips only).
        component_weight = {}
        for comp in non_chips:
            comp_id = comp["comp_id"]
            area_norm = (areas[comp_id] / max_area) if max_area > 0 else 0.0
            degree_norm = (component_degree[comp_id] / max_degree) if max_degree > 0 else 0.0

            round_norm = 0.0
            radius_norm = 0.0
            if component_chip_links[comp_id]:
                round_values = []
                radius_values = []
                for chip_id, pad_idx in component_chip_links[comp_id]:
                    metadata = chip_pad_metadata.get(chip_id, [])
                    if pad_idx < len(metadata):
                        pad_meta = metadata[pad_idx]
                        round_values.append(pad_meta.get("round", 1))
                        radius_values.append(pad_meta.get("radius", 0.0))
                if round_values:
                    round_norm = max(round_values) / max_round if max_round > 0 else 0.0
                if radius_values:
                    radius_norm = max(radius_values) / max_radius if max_radius > 0 else 0.0

            level = levels.get(comp_id, INF_LEVEL)
            if not np.isfinite(level):
                level_bonus = 0.0
            elif max_level > 0:
                level_bonus = 1.0 - (level / (max_level + 1.0))
            else:
                level_bonus = 1.0

            # Linear combination weights tuned to emphasise chip proximity.
            weight = (
                2.0 * area_norm
                + 2.0 * degree_norm
                + 3.0 * round_norm
                + 1.0 * radius_norm
                + 3.0 * level_bonus
            )
            component_weight[comp_id] = weight

        # Provide weights for chips to order them (area + degree preference).
        chip_weight = {}
        for chip in chips:
            cid = chip["comp_id"]
            area_norm = (areas[cid] / max_area) if max_area > 0 else 0.0
            degree_norm = (component_degree[cid] / max_degree) if max_degree > 0 else 0.0
            chip_weight[cid] = area_norm + degree_norm

        scheduled = set()
        sequence = []

        def append_component(comp_id):
            if comp_id in scheduled:
                return
            scheduled.add(comp_id)
            sequence.append(id_to_component[comp_id])

        def schedule_branch(component_id):
            append_component(component_id)
            neighbours = [
                n for n in adjacency[component_id]
                if n not in scheduled and id_to_component[n].get("type_id") != 0
            ]
            neighbours.sort(key=lambda nid: (
                levels.get(nid, INF_LEVEL),
                -component_weight.get(nid, 0.0),
                nid,
            ))
            for neighbour in neighbours:
                schedule_branch(neighbour)

        # Ensure deterministic pad ordering helpers.
        def pad_sort_key(meta_entry):
            return meta_entry.get("angle", 0.0)

        chip_ids_sorted = sorted(
            [chip["comp_id"] for chip in chips],
            key=lambda cid: (-chip_weight.get(cid, 0.0), cid),
        )

        for chip_id in chip_ids_sorted:
            append_component(chip_id)

            metadata = chip_pad_metadata.get(chip_id, [])
            if not metadata:
                continue

            pads_sorted = sorted(metadata, key=pad_sort_key)

            # Choose starting pad based on highest connected component weight.
            best_pad_index = 0
            best_pad_score = -float("inf")
            for idx, entry in enumerate(pads_sorted):
                pad_idx = entry["index"]
                connected = chip_pad_connections[chip_id].get(pad_idx, set())
                if not connected:
                    continue
                score = max(component_weight.get(comp_id, 0.0) for comp_id in connected)
                if score > best_pad_score:
                    best_pad_score = score
                    best_pad_index = idx

            pads_order = pads_sorted[best_pad_index:] + pads_sorted[:best_pad_index]

            for pad_entry in pads_order:
                pad_idx = pad_entry["index"]
                connected_components = [
                    comp_id for comp_id in chip_pad_connections[chip_id].get(pad_idx, set())
                    if comp_id not in scheduled
                ]
                connected_components.sort(
                    key=lambda cid: (
                        -component_weight.get(cid, 0.0),
                        levels.get(cid, INF_LEVEL),
                        cid,
                    )
                )
                for comp_id in connected_components:
                    schedule_branch(comp_id)

        # Append any remaining unscheduled components (e.g., disconnected ones).
        if len(scheduled) < len(components):
            remaining = [comp for comp in components if comp["comp_id"] not in scheduled]

            def remaining_key(comp):
                comp_id = comp["comp_id"]
                level_score = levels.get(comp_id, INF_LEVEL)
                weight = component_weight.get(comp_id, 0.0)
                return (level_score, -weight, comp_id)

            for comp in sorted(remaining, key=remaining_key):
                append_component(comp["comp_id"])

        return sequence

    def _check_boundary(self, x, y, w, h):
        if self.boundary_polygon is not None:
            # 使用更精确的几何检查：采样矩形边界上的多个点
            # 检查四个顶点
            vertices = [
                (x, y),
                (x + w, y),
                (x, y + h),
                (x + w, y + h)
            ]
            for vx, vy in vertices:
                if not self._is_point_in_polygon((vx, vy), self.boundary_polygon):
                    return False
            
            # 采样每条边上的中间点（每条边3个点，包括端点）
            num_samples_per_edge = 3
            sample_points = []
            
            # 上边
            for i in range(num_samples_per_edge):
                sx = x + (w * i) / (num_samples_per_edge - 1)
                sample_points.append((sx, y))
            # 右边
            for i in range(1, num_samples_per_edge):  # 跳过重复的顶点
                sy = y + (h * i) / (num_samples_per_edge - 1)
                sample_points.append((x + w, sy))
            # 下边
            for i in range(1, num_samples_per_edge):  # 跳过重复的顶点
                sx = x + w - (w * i) / (num_samples_per_edge - 1)
                sample_points.append((sx, y + h))
            # 左边
            for i in range(1, num_samples_per_edge - 1):  # 跳过重复的顶点
                sy = y + h - (h * i) / (num_samples_per_edge - 1)
                sample_points.append((x, sy))
            
            # 检查所有采样点
            for sx, sy in sample_points:
                if not self._is_point_in_polygon((sx, sy), self.boundary_polygon):
                    return False
            
            return True
        else:
            # 使用mask检查
            if x < 0 or y < 0: 
                return False
            if x + w > self.N or y + h > self.N:
                return False

            region = self.boundary_mask[y:y+h, x:x+w]
            return np.all(region == 1)

    def _check_overlap(self, x, y, w, h, rot):
        # 检查本体重叠
        region = self.view_mask[y:y+h, x:x+w]
        if np.any(region == 1):
            return False
        
        # 获取当前组件旋转后的pads
        _, _, current_pads = self._get_rotated_dims_and_pads(self.current_component, rot)
        current_pads = [(x + px, y + py) for px, py in current_pads]
        
        # 检查pads重叠：pads与已有本体或pads
        for px, py in current_pads:
            px_int, py_int = int(px), int(py)
            if 0 <= px_int < self.N and 0 <= py_int < self.N:
                if self.view_mask[py_int, px_int] == 1 or self.pad_mask[py_int, px_int] == 1:
                    return False
        
        return True

    def _check_spacing(self, x, y, w, h, rot):
        """
        检查间距约束：包括本体间距和引脚间距
        这是高密度PCB布局的核心硬约束
        """
        current_comp = self.current_component
        current_type = current_comp["type_id"]

        # 获取待放置元件的旋转后的pads
        _, _, current_pads = self._get_rotated_dims_and_pads(current_comp, rot)
        current_pads = [(x + px, y + py) for px, py in current_pads]

        # ================ 本体间距检查 ================
        # 计算待放置元件的包围盒
        current_bbox = (x, y, x + w, y + h)

        for placed_x, placed_y, placed_w, placed_h, placed_type, placed_comp_id in self.placed_components:
            # 计算已放置元件的包围盒
            placed_bbox = (placed_x, placed_y, placed_x + placed_w, placed_y + placed_h)

            # 计算两个包围盒之间的最小距离
            min_distance = self._rect_distance(current_bbox, placed_bbox)

            # 获取对应的间距规则
            body_spacing = self.body_spacing_rules.get((current_type, placed_type),
                                                      self.body_spacing_rules.get((placed_type, current_type), 2))

            if min_distance < body_spacing:
                return False

        # ================ 引脚间距检查 ================
        # 收集所有已放置的引脚坐标及其类型信息
        placed_pads_with_types = []
        for comp in self.components:
            if "placed_pads" in comp and comp["placed_pads"]:
                comp_type = comp["type_id"]
                for pad_x, pad_y in comp["placed_pads"]:
                    placed_pads_with_types.append((pad_x, pad_y, comp_type))

        # 检查当前元件的所有引脚与已放置引脚之间的距离
        for current_pad_x, current_pad_y in current_pads:
            for placed_pad_x, placed_pad_y, placed_type in placed_pads_with_types:
                # 计算欧氏距离
                distance = np.sqrt((current_pad_x - placed_pad_x)**2 + (current_pad_y - placed_pad_y)**2)

                # 获取对应的引脚间距规则
                pad_spacing = self.pad_spacing_rules.get((current_type, placed_type),
                                                        self.pad_spacing_rules.get((placed_type, current_type), 1))

                if distance < pad_spacing:
                    return False

        return True

    def _place_component(self, x, y, w, h, rot, comp_id):
        self.current_component_position = (x, y, rot)

        # 更新placed_pads
        comp = self.current_component
        _, _, rotated_pads = self._get_rotated_dims_and_pads(comp, rot)
        comp["placed_pads"] = [(x + px, y + py) for px, py in rotated_pads]

        # 记录已放置组件 (x, y, w, h, type_id, comp_id)
        type_id = comp["type_id"]
        self.placed_components.append((x, y, w, h, type_id, comp_id))

    def _compute_hpwl(self):
        hpwl = 0
        for net in self.G:  
            xs = []
            ys = []
            for (comp_id, pad_idx) in net:
                comp = self.components[comp_id]
                if pad_idx < len(comp["placed_pads"]):
                    px, py = comp["placed_pads"][pad_idx]
                    xs.append(px)
                    ys.append(py)
            if len(xs) > 1:
                hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))

        return hpwl

    def _compute_slw(self):
        """
        计算 Surface Layer Wire Length (SLW)：按照论文定义
        引脚朝最近边界出线且投影重叠的走线长度
        """
        slw = 0

        pad_directions, pad_positions = self._collect_pad_fanouts()
        if not pad_positions:
            return 0

        for net in self.G:
            if len(net) < 2:
                continue

            direction_groups = {0: [], 1: [], 2: [], 3: []}

            for comp_id, pad_idx in net:
                key = (comp_id, pad_idx)
                direction = pad_directions.get(key)
                pos = pad_positions.get(key)
                if direction is None or pos is None:
                    continue
                if direction in (0, 1):  # top/bottom use x coordinate
                    direction_groups[direction].append(pos[0])
                else:  # left/right use y coordinate
                    direction_groups[direction].append(pos[1])

            for coords in direction_groups.values():
                if len(coords) < 2:
                    continue
                coords.sort()
                slw += coords[-1] - coords[0]

        return slw

    def _compute_nslw(self):
        pad_directions, pad_positions = self._collect_pad_fanouts()
        if not pad_positions:
            return 0

        nslw = 0

        for net in self.G:
            if len(net) < 2:
                continue

            entries = []
            for comp_id, pad_idx in net:
                key = (comp_id, pad_idx)
                direction = pad_directions.get(key)
                pos = pad_positions.get(key)
                if direction is None or pos is None:
                    continue
                entries.append((comp_id, pad_idx, direction, pos))

            count = len(entries)
            for i in range(count):
                comp_i, pad_i, dir_i, pos_i = entries[i]
                for j in range(i + 1, count):
                    comp_j, pad_j, dir_j, pos_j = entries[j]

                    if dir_i != dir_j:
                        continue

                    if dir_i in (0, 1):
                        if abs(pos_i[0] - pos_j[0]) > 1.0:
                            continue
                    else:
                        if abs(pos_i[1] - pos_j[1]) > 1.0:
                            continue

                    if not self._segment_clear(pos_i, pos_j, {comp_i, comp_j}):
                        continue

                    nslw += 1

        return nslw

    def _update_view_mask(self):
        return self.view_mask.copy()

    def build_position_mask(self, component):
        pm = np.ones((self.num_rotations, self.N, self.N), dtype=np.float32)

        comp_type = component["type_id"]
        comp_id = component["comp_id"]

        for r in range(self.num_rotations):
            # 获取旋转后的尺寸
            ww, hh, _ = self._get_rotated_dims_and_pads(component, r)

            for y in range(self.N - hh):
                for x in range(self.N - ww):
                    # boundary
                    if not self._check_boundary(x, y, ww, hh):
                        continue
                    # overlap
                    if not self._check_overlap(x, y, ww, hh, r):
                        continue
                    # spacing
                    if not self._check_spacing(x, y, ww, hh, r):
                        continue

                    pm[r, y, x] = 0  # 0 = 可放置

        return pm

    def build_wire_mask(self, component):
        wm = np.zeros((self.num_rotations, self.N, self.N), dtype=np.float32)

        for r in range(self.num_rotations):
            for y in range(self.N):
                for x in range(self.N):
                    # 对每个位置估计 SLW 变化
                    slw_est = self._estimate_slw_if_placed(component, x, y, r)
                    wm[r, y, x] = np.clip(slw_est, -self.hpwl_range, self.hpwl_range)

        return wm

    def _estimate_hpwl_if_placed(self, comp, x, y, r):
        # 不需要完全精确，只需估计增量
        temp_hpwl = 0

        # 获取旋转后的pads
        _, _, rotated_pads = self._get_rotated_dims_and_pads(comp, r)

        # 对 component 所有 nets 做增量估计
        for net in self.G:
            xs = []
            ys = []
            for (comp_id, pad_idx) in net:
                if comp_id == comp["comp_id"]:
                    # 使用预计位置
                    if pad_idx < len(rotated_pads):
                        px, py = rotated_pads[pad_idx]
                        xs.append(x + px)
                        ys.append(y + py)
                else:
                    if pad_idx < len(self.components[comp_id]["placed_pads"]):
                        px, py = self.components[comp_id]["placed_pads"][pad_idx]
                        xs.append(px)
                        ys.append(py)
                    # else skip
            if len(xs) > 1:
                temp_hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))

        return temp_hpwl - self.hpwl   # 增量

    def _estimate_slw_if_placed(self, comp, x, y, r):
        """
        估计放置组件后的SLW增量
        """
        temp_slw = 0

        # 获取旋转后的pads
        _, _, rotated_pads = self._get_rotated_dims_and_pads(comp, r)

        for net in self.G:
            pads = []
            for comp_id, pad_idx in net:
                if comp_id == comp["comp_id"]:
                    # 使用预计位置
                    if pad_idx < len(rotated_pads):
                        px, py = rotated_pads[pad_idx]
                        pads.append((x + px, y + py))
                else:
                    if pad_idx < len(self.components[comp_id]["placed_pads"]):
                        px, py = self.components[comp_id]["placed_pads"][pad_idx]
                        pads.append((px, py))
            if len(pads) > 1:
                # 计算MST长度
                G = nx.Graph()
                for i, p1 in enumerate(pads):
                    for j, p2 in enumerate(pads):
                        if i < j:
                            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                            G.add_edge(i, j, weight=dist)
                mst = nx.minimum_spanning_tree(G)
                temp_slw += sum(d['weight'] for u, v, d in mst.edges(data=True))

        return temp_slw - self.slw  # 增量

    def _finish_illegal(self, reason):
        """处理非法动作"""
        self.illegal = True
        obs = self._construct_observation(False)
        return obs, -1000, True, {"illegal": True, "reason": reason}

    def _construct_observation(self, done):
        """构造观察"""
        if done:
            # 结束时，返回空component
            component_obs = {
                "size": np.zeros((2,), dtype=np.int32),
                "pad": np.zeros((self.max_pads, 2), dtype=np.int32),
                "type": 0
            }
        else:
            component_obs = {
                "size": np.array(self.current_component["size"], dtype=np.int32),
                "pad": np.array(self.current_component["pad_list"], dtype=np.int32),
                "type": int(self.current_component["type_id"])
            }
        
        obs = {
            "view_mask": self.view_mask.copy(),
            "position_mask": self.position_mask.copy(),
            "wire_mask": self.wire_mask.copy(),
            "component": component_obs
        }
        return obs

    def _rebuild_masks_from_placed_components(self):
        """
        根据 placed_components 重新构建 view_mask 和 pad_mask。
        这是单一事实来源的实现，确保 mask 与 placed_components 始终同步。
        """
        # 重置 masks
        self.view_mask = np.zeros((self.N, self.N), dtype=np.int8)
        self.pad_mask = np.zeros((self.N, self.N), dtype=np.int8)
        
        # 遍历所有已放置组件，重新绘制 masks
        for px, py, pw, ph, type_id, comp_id in self.placed_components:
            # 设置 view_mask（组件本体占据区域）
            self.view_mask[py:py+ph, px:px+pw] = 1
            
            # 设置 pad_mask（引脚占据位置）
            comp = self.components[comp_id]
            if "placed_pads" in comp and comp["placed_pads"]:
                for pad_x, pad_y in comp["placed_pads"]:
                    pad_x_int, pad_y_int = int(pad_x), int(pad_y)
                    if 0 <= pad_x_int < self.N and 0 <= pad_y_int < self.N:
                        self.pad_mask[pad_y_int, pad_x_int] = 1
