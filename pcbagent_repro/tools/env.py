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

def is_slw(p1, p2, comp1, comp2, boundary): #判断点是否共x或共y且重叠
    overlap_h = (min(p1[0], p2[0]) <= max(p1[0], p2[0])) and (abs(p1[1]-p2[1]) < 1e-6)#判断水平对齐，可水平直连，两点y完全相等
    overlap_v = (min(p1[1], p2[1]) <= max(p1[1], p2[1])) and (abs(p1[0]-p2[0]) < 1e-6)#判断垂直对齐，两点x完全相等
    if overlap_h or overlap_v:
        return True
    return False

def nslw(nets, placements, comp_index, boundary): #统计可直线对齐的二端网数量
    cnt = 0
    for net in nets:
        if len(net) != 2: continue #只处理二端网（恰好两个端点的net），多端网被跳过，不纳入统计
        (c1, p1), (c2, p2) = net #第一点元件名c1，pad索引p1；第二点元件名c2，pad索引p2
        if c1 not in placements or c2 not in placements: continue
        A = comp_index[c1]; B = comp_index[c2]
        pa = pad_abs(A, placements[c1], p1); pb = pad_abs(B, placements[c2], p2)
        if is_slw(pa, pb, A, B, boundary): cnt += 1
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

def pad_alignment_bonus(N, next_name, next_comp, rot, nets, placements, comp_index, boundary, b_bonus, eps=1e-6):
    import numpy as np
    bonus = np.zeros((N, N), dtype=float)
    xmin, ymin = boundary[0]; xmax, ymax = boundary[2]
    W, H = xmax - xmin, ymax - ymin
    # 1) 收集所有“对端”(oc, op) 以及 next_name 在该 net 上对应的 pad 索引 p_next
    cons = []  # (oc, op, p_next)
    for net in nets:
        # 找到 net 里 next_name 的所有出现位置
        idxs = [i for i, (c, p) in enumerate(net) if c == next_name]
        if not idxs:
            continue
        # 对每个 next_name 的出现，枚举 net 中其它端点
        for i in idxs:
            c_next, p_next = net[i]
            # 对端：所有 j != i 的端点
            for j, (oc, op) in enumerate(net):
                if j == i:
                    continue
                # 只有当对端组件已放置，才有意义
                if oc in placements:
                    cons.append((oc, op, p_next))
    # 若无任何对端已放置，直接返回全 0
    if not cons:
        return bonus
    # 2) 预先把 next_comp 的 p_next 在 rot 下的“局部旋转偏移”缓存，避免重复 rotate_pad
    #    （一个 net 可能多次用到相同 p_next）
    rot_cache = {}  # p_next -> (ox, oy)
    w, h = next_comp["w"], next_comp["h"]
    # 3) 对每个“对端”，把其 pad 绝对坐标求出来
    targets = []  # (x_pad, y_pad, p_next)
    for oc, op, p_next in cons:
        pad_o = pad_abs(comp_index[oc], placements[oc], op)  # 对端 pad 的绝对坐标
        targets.append((pad_o[0], pad_o[1], p_next))
    # 4) 穷举格点（建议使用格心采样）
    for gy in range(N):
        y0 = ymin + (gy + 0.5) * (H / N)
        for gx in range(N):
            x0 = xmin + (gx + 0.5) * (W / N)
            score = 0.0
            # 针对每个“对端 pad”
            for xp, yp, p_next in targets:
                # 取该 p_next 的旋转偏移（局部），再 + (x0, y0) 得到“如果放在该格点时”的绝对坐标
                if p_next not in rot_cache:
                    px, py = next_comp["pads"][p_next]
                    ox, oy = rotate_pad(px, py, w, h, rot)  # 旋转后仍是局部坐标
                    rot_cache[p_next] = (ox, oy)
                else:
                    ox, oy = rot_cache[p_next]

                cx, cy = x0 + ox, y0 + oy
                # 只要同 x 或同 y，即加分
                if abs(cx - xp) < eps or abs(cy - yp) < eps:
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
        return build_state_token(self.N, self.placements, self.comp_index, self.boundary, next_name, c,
                                 self.nets, self.min_spacing, self.hpwl_thresh, self.b_bonus)

    def place(self, name, x, y, rot): #直接把某器件的左下角坐标与朝向写入 placements
        self.placements[name] = {"x":x,"y":y,"rot":rot}

    def score(self):
        wl = hpwl(self.nets, self.placements, self.comp_index)
        slw = nslw(self.nets, self.placements, self.comp_index, self.boundary)
        return wl, slw

    def legal_states(self):
        return check_legality(self.placements, self.comp_index, self.boundary, self.min_spacing)
