import argparse, json, collections
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def rotate_size(w, h, rot):
    return (w, h) if (rot % 180 == 0) else (h, w)

def rotate_pad(px, py, w, h, rot):
    cx, cy = w/2, h/2
    dx, dy = px - cx, py - cy
    rot = rot % 360
    if rot == 0:   return px, py
    if rot == 90:  return cx + dy, cy - dx
    if rot == 180: return cx - dx, cy - dy
    if rot == 270: return cx - dy, cy + dx
    return px, py

def abs_pad(task, layout, comp_name, pad_idx):
    comp = next((c for c in task["components"] if c["name"] == comp_name), None)
    pl = layout.get(comp_name)
    if comp is None or pl is None or not comp.get("pads") or pad_idx >= len(comp["pads"]):
        return None
    w, h = comp["w"], comp["h"]
    px, py = comp["pads"][pad_idx]
    rx, ry = rotate_pad(px, py, w, h, pl["rot"])
    return (pl["x"] + rx, pl["y"] + ry)

def draw_board(ax, boundary):
    xs = [p[0] for p in boundary]; ys = [p[1] for p in boundary]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False)
    ax.add_patch(rect)
    ax.text((xmin+xmax)/2, ymax, "BOARD", ha='center', va='bottom', fontsize=9)

def draw_components(ax, task, layout, draw_pads=True):
    comp_index = {c["name"]: c for c in task["components"]}
    for name, pl in layout.items():
        comp = comp_index.get(name)
        if comp is None:  # skip unknown
            continue
        w, h = rotate_size(comp["w"], comp["h"], pl["rot"])
        ax.add_patch(Rectangle((pl["x"], pl["y"]), w, h, fill=False))
        cx, cy = pl["x"] + w/2, pl["y"] + h/2
        ax.text(cx, cy, f"{name}\nrot={pl['rot']}", ha='center', va='center', fontsize=8)
        if draw_pads and comp.get("pads"):
            for (px, py) in comp["pads"]:
                rx, ry = rotate_pad(px, py, comp["w"], comp["h"], pl["rot"])
                ax.add_patch(Rectangle((pl["x"]+rx-1.5, pl["y"]+ry-1.5), 3, 3, fill=False))

#布线网络的构建
def build_route_grid(boundary, route_grid):
    xs = [p[0] for p in boundary]; ys = [p[1] for p in boundary]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    gw = route_grid
    w = xmax - xmin; h = ymax - ymin
    gh = max(8, int(round(gw * (h / max(1.0, float(w))))))
    cell_w = w / gw; cell_h = h / gh
    return xmin, ymin, xmax, ymax, gw, gh, cell_w, cell_h
#把世界坐标系（rx0，ry0,rw,rh）的矩形转换为网格索引范围[gx0,gy0]到[gx1,gy1]
def rect_to_grid_indices(xmin, ymin, cell_w, cell_h, gw, gh, rx0, ry0, rw, rh):
    gx0 = max(0, int((rx0 - xmin) // cell_w))
    gy0 = max(0, int((ry0 - ymin) // cell_h))
    gx1 = min(gw-1, int(((rx0 + rw) - xmin) // cell_w))
    gy1 = min(gh-1, int(((ry0 + rh) - ymin) // cell_h))
    return gx0, gy0, gx1, gy1
#把器件外包框在四周各膨胀inflate个单位，得到更保守的阻塞区域
def inflate_rect(plx, ply, w, h, inflate):
    return plx - inflate, ply - inflate, w + 2*inflate, h + 2*inflate
#把每个已放置器件的膨胀外包框离散化到网格上，所有格子标记为障碍（放入集合 obs）
def mark_obstacles(obs, boundary, comps, layout, min_spacing, route_grid):
    xmin, ymin, xmax, ymax, gw, gh, cell_w, cell_h = build_route_grid(boundary, route_grid)
    for comp in comps:
        name = comp["name"]
        if name not in layout:
            continue
        pl = layout[name]
        w, h = rotate_size(comp["w"], comp["h"], pl["rot"])
        rx0, ry0, rw, rh = inflate_rect(pl["x"], pl["y"], w, h, min_spacing)
        gx0, gy0, gx1, gy1 = rect_to_grid_indices(xmin, ymin, cell_w, cell_h, gw, gh, rx0, ry0, rw, rh)
        for gx in range(gx0, gx1+1):
            for gy in range(gy0, gy1+1):
                obs.add((gx, gy))
    return xmin, ymin, xmax, ymax, gw, gh, cell_w, cell_h
#把世界坐标 (x,y) 映到网格索引 (gx,gy)，并夹在 [0..gw-1]×[0..gh-1]
def world_to_grid(x, y, xmin, ymin, cell_w, cell_h, gw, gh):
    gx = min(gw-1, max(0, int((x - xmin) // cell_w)))
    gy = min(gh-1, max(0, int((y - ymin) // cell_h)))
    return gx, gy
#把网格索引转换为该格中心点的世界坐标（画线更直观）
def grid_to_world(gx, gy, xmin, ymin, cell_w, cell_h):
    return xmin + (gx + 0.5) * cell_w, ymin + (gy + 0.5) * cell_h

#BFS————在离散网格上，从起点start到终点goal，避开障碍 obs，用最少步数（曼哈顿距离意义下的最短路径）找到一条路径
def lee_route(start, goal, obs, gw, gh):
    if start == goal:
        return [start] #起始点相同，路径就是单点
    Q = collections.deque([start]) #BFS 用的先进先出队列，从start开始
    # 广度优先搜索（BFS，Breadth-First Search）是一种图遍历算法，适用于寻找从起点到终点的最短路径。
    # 其核心思想是从起点开始，逐层向外扩展，直到找到目标节点。BFS通常使用队列作为核心数据结构，并通过标记已访问节点避免重复搜索。
    visited = {start: None}
    dirs = [(1,0),(-1,0),(0,1),(0,-1)] #4邻接（上下左右），决定这是曼哈顿走法，不允许斜线
    while Q: #逐层扩展：每次从队列取出一个网格点 (x,y)，尝试四个相邻格子
        x, y = Q.popleft()
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= gw or ny >= gh: #越界跳过
                continue
            if (nx, ny) in obs:  #障碍跳过
                continue
            if (nx, ny) in visited:  #已访问跳过
                continue
            visited[(nx, ny)] = (x, y) #首次发现新格子，记录它的父节点为当前格 (x,y)
            if (nx, ny) == goal: #果新格子就是 goal，立刻回溯重建路径 从goal往回找父节点，直到回到 start，反转得到
                path = [(nx, ny)]
                cur = (x, y)
                while cur is not None:
                    path.append(cur)
                    cur = visited[cur]
                path.reverse()
                return path
            Q.append((nx, ny))
    return None #队列空了也没到 goal，说明不可达，返回 None

def simplify_path(path): #去掉直线段中的中间点：把BFS得到的格点串压缩成只保留拐点的折线，绘图更清晰、段数更少
    if not path or len(path) <= 2:
        return path or [] #空路径或单点或两点，直接返回
    out = [path[0]] #输出初始化为起点
    dx0 = path[1][0] - path[0][0] #计算第一段的方向增量
    dy0 = path[1][1] - path[0][1] #4 邻接下，这个方向只能是 (±1,0) 或 (0,±1)
    for i in range(1, len(path)-1): #从第二个点到倒数第二个点，逐点比较下一段方向是否与当前方向一致：
        dx = path[i+1][0] - path[i][0] #一致 ⇒ 还在同一条直线段，中间点 path[i] 不需要；
        dy = path[i+1][1] - path[i][1] #不一致 ⇒ 方向发生改变，说明 path[i] 是转折点，应当保留，并更新方向为新的方向 (dx,dy)
        if (dx, dy) != (dx0, dy0):
            out.append(path[i])
            dx0, dy0 = dx, dy
    out.append(path[-1])
    return out
#task：任务 JSON 已解析的字典； layout：布局 JSON 已解析的字典； png_path：输出 PNG 文件路径字符串； route：是否启用 BFS 路由； route_grid：路由网格分辨率
def render(task, layout, png_path, route=False, route_grid=None):
    boundary = task["boundary"]
    xs = [p[0] for p in boundary]; ys = [p[1] for p in boundary]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    #新建matplotlib画布（8x6英寸）
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_board(ax, boundary) #画出板框矩形和标题board
    draw_components(ax, task, layout, draw_pads=True) #话每个器件的外包框和焊盘

    if route:
        obs = set() #障碍格子集合，记录不能走线的格子
        rg = route_grid or max(32, task.get("grid_N", 32)) #X方向网格数
        xmin_g, ymin_g, xmax_g, ymax_g, gw, gh, cell_w, cell_h = mark_obstacles(
            obs, boundary, task["components"], layout, task.get("min_spacing", 0), rg
        )
        for net in task.get("nets", []):
            if len(net) != 2:
                continue
            (c1, p1), (c2, p2) = net
            a = abs_pad(task, layout, c1, p1); b = abs_pad(task, layout, c2, p2)
            if a is None or b is None:
                continue
            s = world_to_grid(a[0], a[1], xmin_g, ymin_g, cell_w, cell_h, gw, gh)
            t = world_to_grid(b[0], b[1], xmin_g, ymin_g, cell_w, cell_h, gw, gh)
            path = lee_route(s, t, obs, gw, gh)
            #不可达则画一条两点间直线，仅可视化，可能穿越器件
            if not path:
                ax.plot([a[0], b[0]], [a[1], b[1]], linewidth=1)
                continue
            #将 BFS 的格点路径做折线简化（只保留拐点），再转世界坐标（取格中心），画出更清爽的轴对齐折线
            sp = simplify_path(path)
            X, Y = [], []
            for (gx, gy) in sp:
                wx, wy = grid_to_world(gx, gy, xmin_g, ymin_g, cell_w, cell_h)
                X.append(wx); Y.append(wy)
            ax.plot(X, Y, linewidth=1.5)
            # optionally mark routed cells as lightly occupied to reduce crossings
            for (gx, gy) in path:
                obs.add((gx, gy))
    else:
        for net in task.get("nets", []):
            if len(net) < 2: 
                continue
            pts = []
            ok = True
            for (cname, pidx) in net:
                pt = abs_pad(task, layout, cname, pidx)
                if pt is None:
                    ok = False; break
                pts.append(pt)
            if not ok or len(pts) < 2:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=1)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin-10, xmax+10); ax.set_ylim(ymin-10, ymax+10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("PCB Layout Rendering")
    fig.savefig(png_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--layout", type=str, required=True)
    ap.add_argument("--png", type=str, required=True)
    ap.add_argument("--route", action="store_true", help="Use Manhattan BFS sketch router")
    ap.add_argument("--route_grid", type=int, default=64, help="Routing grid resolution along X (default 64)")
    args = ap.parse_args()

    task = json.load(open(args.task, "r", encoding="utf-8"))
    layout = json.load(open(args.layout, "r", encoding="utf-8"))
    render(task, layout, args.png, route=args.route, route_grid=args.route_grid)
    print("[render] wrote", args.png)

if __name__ == "__main__":
    main()
#  & "D:\KiCad\9.0\bin\python.exe" "D:\777777\pcbagent_repro\pcbagent_repro\tools\draw_layout.py" `--task   "D:\777777\pcbagent_repro\pcbagent_repro\tasks\sample_task.json" `--layout "D:\777777\pcbagent_repro\pcbagent_repro\out_layout_refined.json" `--png    "D:\777777\pcbagent_repro\pcbagent_repro\results\pretty_refined.png"