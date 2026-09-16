import gymnasium as gym
import numpy as np
import networkx as nx
from typing import List

class PCBPlacementEnv(gym.Env):

    MAX_MST_PADS = 24

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
        # NSLW metric
        self._nslw = 0
        self.prev_nslw = 0
        # score weights (paper uses lambda1=1000, lambda2=0.01 in Eq.2)
        self.lambda1 = 1000.0
        self.lambda2 = 0.01
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

        def _on_segment(pt, a, b, tol=1e-6):
            px, py = pt
            ax, ay = a
            bx, by = b

            cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
            if abs(cross) > tol:
                return False

            dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
            if dot < -tol:
                return False

            length_sq = (bx - ax) ** 2 + (by - ay) ** 2
            if dot > length_sq + tol:
                return False

            return True

        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]

            if _on_segment(point, (p1x, p1y), (p2x, p2y)):
                return True

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


    def _segment_intersects_rect(self, p1, p2, rect):
        """
        Check if segment p1-p2 intersects axis-aligned rectangle rect=(x1,y1,x2,y2).
        """
        (x1, y1) = p1
        (x2, y2) = p2
        rx1, ry1, rx2, ry2 = rect

        # Liang-Barsky algorithm for segment-rect intersection
        dx = x2 - x1
        dy = y2 - y1

        p = [-dx, dx, -dy, dy]
        q = [x1 - rx1, rx2 - x1, y1 - ry1, ry2 - y1]

        u1 = 0.0
        u2 = 1.0
        for pi, qi in zip(p, q):
            if abs(pi) < 1e-9:
                if qi < 0:
                    return False
                else:
                    continue
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
        return True


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
        self._nslw = 0
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
        self.prev_nslw = getattr(self, '_nslw', 0)
        self._nslw = getattr(self, '_nslw', 0)

        # compute score per paper: score = lambda1 * (1/HPWL) + lambda2 * NSLW
        def _score(hpwl_val, nslw_val):
            hpwl_term = 0.0
            if hpwl_val > 0:
                hpwl_term = self.lambda1 * (1.0 / float(hpwl_val))
            # if hpwl_val == 0, keep hpwl_term as large const? avoid div0 by skipping
            return hpwl_term + self.lambda2 * float(nslw_val)

        prev_score = _score(self.prev_hpwl if self.prev_hpwl is not None else self.hpwl, self.prev_nslw)
        cur_score = _score(self.hpwl, self._nslw)
        reward = cur_score - prev_score
        # 调整HPWL和SLW权重，避免过强信号导致不稳定
        hpwl_reward = (self.prev_hpwl - self.hpwl) * 1.5
        slw_reward = (self.prev_slw - self.slw) * 1.2
        reward = hpwl_reward + slw_reward

        # 是否结束
        self.current_index += 1
        info = {"hpwl": self.hpwl, "slw": self.slw, "nslw": int(self._nslw), "score": cur_score}
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
        from matplotlib.patches import Polygon, FancyBboxPatch, Circle

        board_polygon = self.boundary_polygon
        if board_polygon:
            board_poly_arr = np.array(board_polygon)
            min_x = float(board_poly_arr[:, 0].min())
            max_x = float(board_poly_arr[:, 0].max())
            min_y = float(board_poly_arr[:, 1].min())
            max_y = float(board_poly_arr[:, 1].max())
        else:
            min_x, min_y = 0.0, 0.0
            max_x, max_y = float(self.N), float(self.N)
            board_polygon = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y)
            ]

        extent_x = max_x - min_x
        extent_y = max_y - min_y
        margin = max(1.5, 0.05 * max(extent_x, extent_y))

        fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
        ax.set_facecolor('#0f1414')
        ax.set_aspect('equal')

        board_face = '#1b5e20'
        board_edge = '#0b3d0b'
        board_patch = Polygon(board_polygon, closed=True, facecolor=board_face, edgecolor=board_edge, linewidth=2.0, zorder=1)
        ax.add_patch(board_patch)

        modules = None
        if hasattr(self, '_source_board') and getattr(self, '_source_board'):
            modules = self._source_board.get('modules')

        # 绘制布线（半透明淡黄色线条）
        net_color = '#ffeb99'
        for net in self.G:
            pads = []
            for comp_id, pad_idx in net:
                if comp_id >= len(self.components):
                    continue
                comp = self.components[comp_id]
                placed = comp.get("placed_pads") or []
                if pad_idx < len(placed):
                    pads.append(placed[pad_idx])
            if len(pads) < 2:
                continue
            for i in range(len(pads)):
                for j in range(i + 1, len(pads)):
                    x1, y1 = pads[i]
                    x2, y2 = pads[j]
                    ax.plot([x1, x2], [y1, y2], color=net_color, linewidth=0.6, alpha=0.35, zorder=2)

        # 绘制已放置组件
        color_map = plt.get_cmap('tab20')
        for px, py, pw, ph, type_id, comp_id in self.placed_components:
            body_color = color_map(type_id % color_map.N)
            body_patch = FancyBboxPatch(
                (px, py),
                pw,
                ph,
                boxstyle="round,pad=0.3",
                facecolor=body_color,
                edgecolor='#0a0a0a',
                linewidth=1.5,
                alpha=0.78,
                zorder=3
            )
            ax.add_patch(body_patch)

            pads = []
            if comp_id < len(self.components):
                pads = self.components[comp_id].get("placed_pads") or []

            for pad_x, pad_y in pads:
                pad_patch = Circle(
                    (pad_x, pad_y),
                    radius=0.6,
                    facecolor='#d6a447',
                    edgecolor='#8c5d1a',
                    linewidth=0.6,
                    zorder=4
                )
                ax.add_patch(pad_patch)

            label = None
            if modules and comp_id < len(modules):
                label = modules[comp_id].get('reference') or modules[comp_id].get('footprint')
            if not label:
                label = f'C{comp_id}'
            ax.text(
                px + pw / 2.0,
                py + ph / 2.0,
                label,
                ha='center',
                va='center',
                fontsize=6,
                fontweight='bold',
                color='#0b0b0b',
                zorder=5
            )

        # 高亮当前待放置元件
        if self.current_component_position is not None and self.current_index < self.num_components:
            cx, cy, rot = self.current_component_position
            w, h, rotated_pads = self._get_rotated_dims_and_pads(self.current_component, rot)
            highlight = FancyBboxPatch(
                (cx, cy),
                w,
                h,
                boxstyle="round,pad=0.25",
                facecolor='none',
                edgecolor='#ff7043',
                linewidth=1.6,
                linestyle='--',
                zorder=6
            )
            ax.add_patch(highlight)
            for pad_x, pad_y in rotated_pads:
                pad_patch = Circle(
                    (cx + pad_x, cy + pad_y),
                    radius=0.55,
                    facecolor='#ffcc80',
                    edgecolor='#bf6f31',
                    linewidth=0.6,
                    zorder=7
                )
                ax.add_patch(pad_patch)

        # 坐标与刻度显示
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
        ax.invert_yaxis()

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#344')
        ax.spines['bottom'].set_color('#344')

        xticks = np.linspace(min_x, max_x, num=6)
        yticks = np.linspace(min_y, max_y, num=6)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        xlabels = [f"{tick:.1f}" for tick in (xticks - min_x)]
        ylabels = [f"{tick:.1f}" for tick in (yticks - min_y)]

        if hasattr(self, '_board_size_mm') and getattr(self, '_board_size_mm'):
            board_mm_x, board_mm_y = self._board_size_mm
            width_grid = max(max_x - min_x, 1e-6)
            height_grid = max(max_y - min_y, 1e-6)
            xlabels = [f"{(tick - min_x) * (board_mm_x / width_grid):.1f}" for tick in xticks]
            ylabels = [f"{(tick - min_y) * (board_mm_y / height_grid):.1f}" for tick in yticks]
            ax.set_xlabel('mm', color='#d0d7d0')
            ax.set_ylabel('mm', color='#d0d7d0')
        else:
            ax.set_xlabel('grid', color='#d0d7d0')
            ax.set_ylabel('grid', color='#d0d7d0')

        ax.set_xticklabels(xlabels, color='#d0d7d0')
        ax.set_yticklabels(ylabels, color='#d0d7d0')
        ax.tick_params(colors='#d0d7d0', labelsize=7)

        ax.set_title(f"PCB Placement – HPWL {self.hpwl:.1f} / SLW {self.slw:.1f}", color='#e8f5e9', fontsize=10)

        plt.tight_layout()

        if mode == "rgb_array":
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
            buf.seek(0)

            from PIL import Image
            img = Image.open(buf)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            canvas = np.array(img)

            plt.close(fig)
            return canvas
        elif mode == "human":
            plt.savefig('matplotlib_render.png', dpi=160, bbox_inches='tight')
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
        """
        按照芯片优先 + 权重排序策略生成放置序列，提升高密度布板鲁棒性。
        """
        if not components:
            return []

        id_to_comp = {comp["comp_id"]: comp for comp in components}
        pad_counts = {comp_id: len(comp.get("pad_list", [])) for comp_id, comp in id_to_comp.items()}
        areas = {comp_id: max(1, int(comp.get("size", [1, 1])[0]) * int(comp.get("size", [1, 1])[1])) for comp_id, comp in id_to_comp.items()}

        adjacency = {comp_id: set() for comp_id in id_to_comp.keys()}
        for net in self.netlist or []:
            members = [comp_id for comp_id, _ in net if comp_id in adjacency]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    adjacency[a].add(b)
                    adjacency[b].add(a)

        max_pads = max(pad_counts.values()) if pad_counts else 1
        max_area = max(areas.values()) if areas else 1
        max_degree = max((len(neigh) for neigh in adjacency.values()), default=1)

        def _weight(comp_id: int) -> float:
            pad_term = pad_counts.get(comp_id, 0) / max(max_pads, 1)
            area_term = areas.get(comp_id, 1) / max(max_area, 1)
            degree_term = len(adjacency.get(comp_id, [])) / max(max_degree, 1)
            return 1.5 * pad_term + 1.0 * area_term + 1.2 * degree_term

        chip_candidates = [comp_id for comp_id, pads in pad_counts.items() if pads == max_pads]
        if not chip_candidates:
            chip_candidates = list(id_to_comp.keys())

        chip_candidates.sort(key=lambda cid: (_weight(cid), areas.get(cid, 0)), reverse=True)

        ordered_ids = []
        visited = set()
        queue = list(chip_candidates)

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            ordered_ids.append(current)

            neighbors = sorted(
                (n for n in adjacency.get(current, set()) if n not in visited),
                key=_weight,
                reverse=True,
            )
            queue.extend(neighbors)

        remaining = sorted((cid for cid in id_to_comp.keys() if cid not in visited), key=_weight, reverse=True)
        ordered_ids.extend(remaining)

        return [id_to_comp[cid] for cid in ordered_ids]

    def _check_boundary(self, x, y, w, h):
        if self.boundary_polygon is not None:
            if w <= 0 or h <= 0:
                return False

            half = 0.5

            def _inside(px: float, py: float) -> bool:
                return self._is_point_in_polygon((px, py), self.boundary_polygon)

            # 检查使用网格中心的四个角点
            corners = [
                (x + half, y + half),
                (x + w - half, y + half),
                (x + half, y + h - half),
                (x + w - half, y + h - half),
            ]
            for vx, vy in corners:
                if not _inside(vx, vy):
                    return False

            # 采样四条边，确保矩形完全位于多边形内部
            num_samples_per_edge = 5

            def _edge_offsets(length: int) -> List[float]:
                if length <= 1:
                    return [0.0]
                step = (length - 1) / (num_samples_per_edge - 1)
                return [step * i for i in range(num_samples_per_edge)]

            x_offsets = _edge_offsets(w)
            y_offsets = _edge_offsets(h)

            for offset in x_offsets:
                if not _inside(x + offset + half, y + half):
                    return False
                if not _inside(x + offset + half, y + h - half):
                    return False

            for offset in y_offsets:
                if not _inside(x + half, y + offset + half):
                    return False
                if not _inside(x + w - half, y + offset + half):
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
        """
        计算半周长布线长度 (Half-Perimeter Wire Length)

        修改逻辑：即使net中只有一个pad被放置，也要考虑其对布线长度的影响
        为部分放置的net提供有意义的奖励信号
        """
        hpwl = 0.0
        for net in self.G:
            xs = []
            ys = []

            # 收集所有已放置的pad
            for (comp_id, pad_idx) in net:
                comp = self.components[comp_id]
                if pad_idx < len(comp.get("placed_pads", [])):
                    px, py = comp["placed_pads"][pad_idx]
                    xs.append(px)
                    ys.append(py)

            if len(xs) == 0:
                continue
            elif len(xs) == 1:
                # 单个pad：使用其到中心的L1距离作为估计（与论文一致）
                center_x, center_y = (self.N - 1) / 2.0, (self.N - 1) / 2.0
                hpwl += abs(xs[0] - center_x) + abs(ys[0] - center_y)
            else:
                # 多个pad：传统HPWL
                hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))

        return float(hpwl)

    def _compute_slw(self):
        """
        计算 Surface Layer Wire Length (SLW)：按照论文定义
        引脚朝最近边界出线且投影重叠的走线长度
        """
        slw = 0.0
        nslw = 0

        # helper: nearest edge direction for a pad coordinate
        def _pad_direction(px, py):
            # distances to edges (top=0, bottom=1, left=2, right=3)
            d_top = py
            d_bottom = (self.N - 1) - py
            d_left = px
            d_right = (self.N - 1) - px
            distances = [d_top, d_bottom, d_left, d_right]
            return int(np.argmin(distances))

        # helper: check if segment intersects any placed component bbox (excluding the two comps themselves)
        def _segment_intersects_any_bbox(p1, p2, ignore_comp_ids):
            x1, y1 = p1
            x2, y2 = p2
            for px, py, pw, ph, type_id, comp_id in self.placed_components:
                if comp_id in ignore_comp_ids:
                    continue
                # bbox corners
                bx1, by1 = px, py
                bx2, by2 = px + pw, py + ph
                # check segment vs rect intersection
                if self._segment_intersects_rect((x1, y1), (x2, y2), (bx1, by1, bx2, by2)):
                    return True
            return False

        # iterate nets and count SLW pairs
        for net in self.G:
            # collect placed pads with (comp_id, pad_idx, x, y)
            placed = []
            for comp_id, pad_idx in net:
                comp = self.components[comp_id]
                if pad_idx < len(comp.get("placed_pads", [])):
                    px, py = comp["placed_pads"][pad_idx]
                    placed.append((comp_id, pad_idx, float(px), float(py)))

            L = len(placed)
            if L < 2:
                continue

            # check every unordered pair
            for i in range(L):
                for j in range(i + 1, L):
                    a = placed[i]
                    b = placed[j]
                    comp_a, pad_a, ax, ay = a
                    comp_b, pad_b, bx, by = b

                    dir_a = _pad_direction(ax, ay)
                    dir_b = _pad_direction(bx, by)

                    # Condition (1): fan-out directions are towards one of the two edges closest to their pads
                    # (we already computed nearest edge; accept it)

                    # Condition (2): segment between pads does not intersect other components
                    if _segment_intersects_any_bbox((ax, ay), (bx, by), ignore_comp_ids={comp_a, comp_b}):
                        continue

                    # Condition (3): projections overlap on axis orthogonal to fan-out direction
                    # map directions to orthogonal axis: top/bottom -> x-axis, left/right -> y-axis
                    def _proj_overlap(a_coord, b_coord, axis):
                        # axis 0 = x, 1 = y
                        if axis == 0:
                            a1, a2 = a_coord[0], b_coord[0]
                        else:
                            a1, a2 = a_coord[1], b_coord[1]
                        # here projection of a point is just its coordinate; overlap for points means equality tolerance
                        # but we consider pads as points; to match paper idea, require interval overlap of pads' projection
                        # since pads are points, we treat as overlap if coords are within 1 unit
                        return abs(a1 - a2) <= 1.0

                    # choose axis orthogonal to the outward direction of either pad; require overlap for either mapping
                    axis_a = 0 if dir_a in (0, 1) else 1
                    axis_b = 0 if dir_b in (0, 1) else 1

                    # Require that the projections overlap on at least one of the pads' orthogonal axes
                    overlap = _proj_overlap((ax, ay), (bx, by), axis_a) or _proj_overlap((ax, ay), (bx, by), axis_b)
                    if not overlap:
                        continue

                    # Passed all three conditions: count as SLW
                    nslw += 1

                    # For SLW length metric, add projection span on orthogonal axis (use average of axes spans)
                    if axis_a == 0:
                        slw += abs(ax - bx)
                    else:
                        slw += abs(ay - by)

        # store NSLW for external access
        self._nslw = int(nslw)
        return float(slw)

    def _update_view_mask(self):
        return self.view_mask.copy()

    def build_position_mask(self, component):
        pm = np.ones((self.num_rotations, self.N, self.N), dtype=np.float32)

        comp_type = component["type_id"]
        comp_id = component["comp_id"]

        for r in range(self.num_rotations):
            # 获取旋转后的尺寸
            ww, hh, _ = self._get_rotated_dims_and_pads(component, r)

            max_y = max(self.N - hh + 1, 0)
            max_x = max(self.N - ww + 1, 0)

            for y in range(max_y):
                for x in range(max_x):
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
        # 临时禁用SLW计算以提高性能
        wm = np.zeros((self.num_rotations, self.N, self.N), dtype=np.float32)
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
        temp_slw = 0.0

        # helper similar to _compute_slw but only computing contribution when comp's pads participate
        def _pad_direction(px, py):
            d_top = py
            d_bottom = (self.N - 1) - py
            d_left = px
            d_right = (self.N - 1) - px
            distances = [d_top, d_bottom, d_left, d_right]
            return int(np.argmin(distances))

        def _segment_intersects_any_bbox_local(p1, p2, ignore_comp_ids):
            x1, y1 = p1
            x2, y2 = p2
            for px, py, pw, ph, type_id, comp_id in self.placed_components:
                if comp_id in ignore_comp_ids:
                    continue
                bx1, by1 = px, py
                bx2, by2 = px + pw, py + ph
                if self._segment_intersects_rect((x1, y1), (x2, y2), (bx1, by1, bx2, by2)):
                    return True
            return False

        _, _, rotated_pads = self._get_rotated_dims_and_pads(comp, r)

        for net in self.G:
            placed = []
            comp_in_net = False
            for comp_id, pad_idx in net:
                if comp_id == comp["comp_id"]:
                    if pad_idx < len(rotated_pads):
                        px, py = rotated_pads[pad_idx]
                        placed.append((comp_id, pad_idx, float(x + px), float(y + py)))
                        comp_in_net = True
                else:
                    if pad_idx < len(self.components[comp_id].get("placed_pads", [])):
                        px, py = self.components[comp_id]["placed_pads"][pad_idx]
                        placed.append((comp_id, pad_idx, float(px), float(py)))

            if not comp_in_net:
                continue

            L = len(placed)
            if L < 2:
                continue

            for i in range(L):
                for j in range(i + 1, L):
                    a = placed[i]
                    b = placed[j]
                    comp_a, pad_a, ax, ay = a
                    comp_b, pad_b, bx, by = b

                    dir_a = _pad_direction(ax, ay)
                    dir_b = _pad_direction(bx, by)

                    if _segment_intersects_any_bbox_local((ax, ay), (bx, by), ignore_comp_ids={comp_a, comp_b}):
                        continue

                    axis_a = 0 if dir_a in (0, 1) else 1
                    axis_b = 0 if dir_b in (0, 1) else 1

                    overlap = (abs((ax if axis_a == 0 else ay) - (bx if axis_a == 0 else by)) <= 1.0) or \
                              (abs((ax if axis_b == 0 else ay) - (bx if axis_b == 0 else by)) <= 1.0)
                    if not overlap:
                        continue

                    # add projection span
                    if axis_a == 0:
                        temp_slw += abs(ax - bx)
                    else:
                        temp_slw += abs(ay - by)

        return temp_slw - self.slw

    def _finish_illegal(self, reason):
        """处理非法动作"""
        self.illegal = True
        obs = self._construct_observation(False)
        # 降低非法动作惩罚，从-1000降低到-50，使其与奖励重塑兼容
        return obs, -50, True, {"illegal": True, "reason": reason}

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
