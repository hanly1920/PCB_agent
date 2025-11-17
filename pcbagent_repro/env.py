import math, numpy as np, json, copy, random
from typing import List, Tuple, Dict

#几何工具函数
def rect_intersect(ax, ay, aw, ah, bx, by, bw, bh) -> bool: #轴对齐矩形相交判定，<=表示边界接触不算相交
    return not (ax+aw <= bx or bx+bw <= ax or ay+ah <= by or by+bh <= ay)

def rect_within_boundary(x, y, w, h, boundary): #元件框是否完全落在边界内
    xmin, ymin = boundary[0]
    xmax, ymax = boundary[2]
    return x >= xmin and y >= ymin and (x+w) <= xmax and (y+h) <= ymax

def manhattan(p, q): #曼哈顿距离
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def rotate_size(w, h, rot): #旋转后的宽高,90/270度交换宽高
    if rot % 180 == 0: return w, h
    return h, w

def rotate_pad(px, py, w, h, rot): #px, py 是 pad 在元件局部坐标系中的位置
    cx, cy = w/2, h/2
    dx, dy = px - cx, py - cy
    rot = rot % 360
    if rot == 0:   return px, py
    if rot == 90:  return cx + dy, cy - dx #90°——把 (dx,dy) 变为 (dy, -dx)，顺时针 90°旋转
    if rot == 180: return cx - dx, cy - dy #180°——把 (dx,dy) 变为 (-dx, -dy)
    if rot == 270: return cx - dy, cy + dx #270°——把 (dx,dy) 变为 (-dy, dx)
    return px, py

def hpwl(nets, placements, comp_index): #计算半周长线长
    wl = 0.0
    for net in nets:
        points = []
        ok = True
        for (cname, pidx) in net:
            if cname not in placements:
                ok = False; break
            comp = comp_index[cname] # 获取元件信息
            x, y, rot = placements[cname]["x"], placements[cname]["y"], placements[cname]["rot"]
            w, h = comp["w"], comp["h"]
            px, py = comp["pads"][pidx]
            pxr, pyr = rotate_pad(px, py, w, h, rot)
            points.append((x+pxr, y+pyr)) #计算pad的绝对坐标
        if ok and points:
            xs = [p[0] for p in points]; ys = [p[1] for p in points]
            wl += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return wl

def is_slw(p1, p2, comp1, comp2, boundary, max_len=None):
    dx = abs(p1[0] - p2[0]) #p1是第一个pad的绝对坐标，p2是第二个pad的绝对坐标
    dy = abs(p1[1] - p2[1])
    overlap_h = (dy < 1e-6)
    overlap_v = (dx < 1e-6)
    if not (overlap_h or overlap_v):
        return False
    length = dx + dy  # 曼哈顿
    if max_len is not None and length > max_len:
        return False   # 太长的“直线”不加分
    return True

def nslw(nets, placements, comp_index, boundary): #统计可直线对齐的二端网数量
    # 在这里根据板子尺寸算一个最大长度阈值
    xmin, ymin = boundary[0]
    xmax, ymax = boundary[2]
    board_w = xmax - xmin
    board_h = ymax - ymin
    # 比如：超过板子“周长”的 1/4 就不算好直线，你可以自己调这个 0.25
    max_len = 0.25 * (board_w + board_h)

    cnt = 0
    for net in nets:
        if len(net) != 2: continue #只处理二端网（恰好两个端点的net），多端网被跳过，不纳入统计
        (c1, p1), (c2, p2) = net #第一点元件名c1，pad索引p1；第二点元件名c2，pad索引p2
        if c1 not in placements or c2 not in placements: continue
        A = comp_index[c1]; B = comp_index[c2]
        pa = pad_abs(A, placements[c1], p1); pb = pad_abs(B, placements[c2], p2)
        if is_slw(pa, pb, A, B, boundary, max_len=max_len): cnt += 1
    return cnt

def pad_abs(comp, place, pad_idx): #计算pad的绝对坐标
    x, y, rot = place["x"], place["y"], place["rot"]
    w, h = comp["w"], comp["h"]
    px, py = comp["pads"][pad_idx] # 该 pad 的局部坐标（以左下为原点）
    pxr, pyr = rotate_pad(px, py, w, h, rot)
    return (x+pxr, y+pyr)

def view_mask(N, placements, comp_index, boundary):#生成已放置元件的占用掩码
    m = np.zeros((N,N), dtype=np.uint8) #创建一个N×N的零矩阵
    xmin,ymin = boundary[0]; xmax,ymax = boundary[2]
    W = xmax-xmin; H = ymax-ymin
    for cname, pl in placements.items():
        comp = comp_index[cname]
        w,h = rotate_size(comp["w"], comp["h"], pl["rot"])
        gx0 = int((pl["x"]-xmin)/W*N); gy0 = int((pl["y"]-ymin)/H*N)
        gx1 = int((pl["x"]+w-xmin)/W*N); gy1 = int((pl["y"]+h-ymin)/H*N)
        m[gy0:gy1, gx0:gx1] = 1 #将矩形区域切片标为1
    return m

def legal_position_mask(N, next_comp, placements, comp_index, boundary, min_spacing, rot): #生成下一个元件的合法放置位置掩码
    m = np.zeros((N,N), dtype=np.uint8)
    xmin,ymin = boundary[0]; xmax,ymax = boundary[2]
    W = xmax-xmin; H = ymax-ymin
    w,h = rotate_size(next_comp["w"], next_comp["h"], rot)
    for gy in range(N):
        for gx in range(N):
            x = xmin + gx * (W/N)
            y = ymin + gy * (H/N)
            if not rect_within_boundary(x, y, w, h, boundary):
                m[gy,gx] = 1; continue
            bad = False
            for cname, pl in placements.items():
                c = comp_index[cname]
                ww,hh = rotate_size(c["w"], c["h"], pl["rot"])
                if rect_intersect(x-min_spacing, y-min_spacing, w+2*min_spacing, h+2*min_spacing,
                                  pl["x"], pl["y"], ww, hh): #用min_spacing 扩展检测
                    bad = True; break
            if bad: m[gy,gx] = 1
    return m

def wire_mask(N, next_name, next_comp, rot, nets, placements, comp_index, boundary, hpwl_thresh): #生成下一个元件的连线代价掩码
    wm = np.full((N,N), fill_value=np.inf, dtype=float)
    xmin,ymin = boundary[0]; xmax,ymax = boundary[2]
    W = xmax-xmin; H = ymax-ymin
    pre = hpwl(nets, placements, comp_index)
    for gy in range(N):
        for gx in range(N):
            x = xmin + gx * (W/N); y = ymin + gy * (H/N)
            tmp = placements.copy()
            tmp[next_name] = {"x":x,"y":y,"rot":rot}
            post = hpwl(nets, tmp, comp_index)
            delta = max(0.0, post - pre) #只惩罚增加
            wm[gy,gx] = delta if delta <= hpwl_thresh else 1e9 #若 delta 大于门槛 hpwl_thresh，写入一个大惩罚 1e9（表示几乎不可取）
    return wm

#给待放置器件在朝向rot下、落在每个网格格心时的对齐奖励打分，只要该器件某个将要用到的pad的绝对坐标，与某个已放置器件的某个pad的绝对坐标在x或y上对齐，就在那个格子上加b_bonus分，最后返回一个NxN的bonus矩阵
def pad_alignment_bonus( N, next_name, next_comp, rot, nets, placements, comp_index, boundary,b_bonus, slw_tol_cells=0.5):
    bonus = np.zeros((N, N), dtype=np.float32)
    xmin, ymin = boundary[0]; xmax, ymax = boundary[2]
    W, H = xmax - xmin, ymax - ymin
    dx, dy = W / N, H / N
    tol_x = dx * slw_tol_cells
    tol_y = dy * slw_tol_cells
    # 1) 收集已放置对端
    cons = []
    for net in nets:
        idxs = [i for i, (c, _) in enumerate(net) if c == next_name]
        for i in idxs:
            _, p_next = net[i]
            for j, (oc, op) in enumerate(net):
                if j == i: continue
                if oc in placements:
                    cons.append((oc, op, p_next))
    if not cons:
        return bonus
    # 2) 缓存 next 的旋转 pad 偏移；3) 生成对端绝对坐标
    rot_cache = {}
    w, h = next_comp["w"], next_comp["h"]
    targets = []
    for oc, op, p_next in cons:
        xp, yp = pad_abs(comp_index[oc], placements[oc], op)
        targets.append((xp, yp, p_next))
    # 4) 穷举格心 + 带容差的对齐判定
    for gy in range(N):
        y0 = ymin + (gy + 0.5) * dy
        for gx in range(N):
            x0 = xmin + (gx + 0.5) * dx
            score = 0.0
            for xp, yp, p_next in targets:
                if p_next not in rot_cache:
                    px, py = next_comp["pads"][p_next]
                    rot_cache[p_next] = rotate_pad(px, py, w, h, rot)
                ox, oy = rot_cache[p_next]
                cx, cy = x0 + ox, y0 + oy
                # 容差内算命中
                if (abs(cx - xp) <= tol_x) or (abs(cy - yp) <= tol_y):
                    score += b_bonus
            if score > 0:
                bonus[gy, gx] += score
    return bonus


def build_state_token(N, placements, comp_index, boundary, next_name, next_comp, nets, min_spacing, hpwl_thresh, b_bonus):#对rot四个朝向分别生成三张图
    tokens = {}
    for rot in [0,90,180,270]:
        pm = legal_position_mask(N, next_comp, placements, comp_index, boundary, min_spacing, rot)
        wm = wire_mask(N, next_name, next_comp, rot, nets, placements, comp_index, boundary, hpwl_thresh)
        bonus = pad_alignment_bonus(N, next_name, next_comp, rot, nets, placements, comp_index, boundary, b_bonus)
        tokens[rot] = dict(position=pm, wire=wm, bonus=bonus)
    return tokens

def check_legality(placements, comp_index, boundary, min_spacing): #整体合法性检查
    boundary_bad = []
    spacing_bad = []
    for name, pl in placements.items():
        c = comp_index[name]
        w,h = rotate_size(c["w"], c["h"], pl["rot"])
        if not rect_within_boundary(pl["x"], pl["y"], w, h, boundary):
            boundary_bad.append(name)
    items = list(placements.items())
    for i in range(len(items)):
        n1, p1 = items[i]
        c1 = comp_index[n1]; w1,h1 = rotate_size(c1["w"], c1["h"], p1["rot"])
        for j in range(i+1,len(items)):
            n2, p2 = items[j]
            c2 = comp_index[n2]; w2,h2 = rotate_size(c2["w"], c2["h"], p2["rot"])
            if rect_intersect(p1["x"]-min_spacing, p1["y"]-min_spacing, w1+2*min_spacing, h1+2*min_spacing,
                              p2["x"], p2["y"], w2, h2):
                spacing_bad.append((n1, n2))
    boundary_ok = (len(boundary_bad) == 0)
    spacing_ok = (len(spacing_bad) == 0)
    all_ok = boundary_ok and spacing_ok
    return all_ok, boundary_ok, spacing_ok, boundary_bad, spacing_bad

class PCBEnv:
    def __init__(self, task): #从task里抽取环境参数与数据
        self.task = task
        self.N = task["grid_N"] #网格尺寸N×N
        self.boundary = task["boundary"] #板子工作区域 矩形
        self.min_spacing = task["min_spacing"] #元件间最小间距
        self.hpwl_thresh = task["hpwl_thresh"] #连线代价惩罚阈值
        self.b_bonus = task["b_bonus"] #pad对齐奖励
        self.comp_index = {c["name"]: c for c in task["components"]} #元件名到元件信息的映射
        self.nets = task["nets"] #网络连接关系的列表
        self.reset()

    def reset(self):
        self.placements = {}
        return self.placements

    def state_token(self, next_name):
        c = self.comp_index[next_name]
        tokens = build_state_token(
            self.N, self.placements, self.comp_index, self.boundary,
            next_name, c, self.nets, self.min_spacing, self.hpwl_thresh, self.b_bonus
        )

        # 允许从 task 里配置；没配就用默认
        slw_tol = getattr(self, "slw_tol_cells", 0.5)
        b_bonus = getattr(self, "b_bonus", 0.1)
        hp = getattr(self, "hpwl_thresh", 10.0)  # 容错：没配就用 10.0
        # 给每个 rot 补 bonus（基于“对端 pad 共线 + 带容差”）
        for rot, t in tokens.items():
            position = np.asarray(t["position"], dtype=np.float32)  # 1=非法, 0=合法

            # === 1) wire 归一化，关键：剪掉 inf / 1e9，把数值压到 [0,1] ===
            wire = np.asarray(t["wire"], dtype=np.float32)
            # 把 inf 替换成一个大值
            wire[~np.isfinite(wire)] = hp
            # 大于 hp 的（比如 1e9）直接截到 hp
            wire = np.clip(wire, 0.0, hp)
            # 映射到 [0,1]
            wire = wire / max(hp, 1e-6)

            bonus = pad_alignment_bonus(
                N=self.N,
                next_name=next_name,
                next_comp=c,
                rot=rot,
                nets=self.nets,
                placements=self.placements,
                comp_index=self.comp_index,
                boundary=self.boundary,
                b_bonus=b_bonus,
                slw_tol_cells=slw_tol,  # ← 关键：容差（按“格子数”）
            )
            # 非法格奖励清零，避免被“奖励”抬高
            bonus[position >= 0.5] = 0.0
            t["position"] = position
            t["wire"] = wire.astype(np.float32)
            t["bonus"] = bonus.astype(np.float32)
        return tokens

    def place(self, name, x, y, rot): #直接把某器件的左下角坐标与朝向写入 placements
        self.placements[name] = {"x":x,"y":y,"rot":rot}

    def score(self):
        wl = hpwl(self.nets, self.placements, self.comp_index)
        slw = nslw(self.nets, self.placements, self.comp_index, self.boundary)
        return wl, slw

    def legal_states(self):
        return check_legality(self.placements, self.comp_index, self.boundary, self.min_spacing)
