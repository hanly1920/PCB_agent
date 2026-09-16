import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# ===== 需要改的只有这一行：你的 data 根目录 =====
ROOT_DIR = r"E:\77777\my(1)\my\data"
# 所有图片输出到这个文件夹里
OUTPUT_DIR = os.path.join(ROOT_DIR, "plots")


def load_task_and_layout(traj_dir: str):
    """
    根据文件夹名自动推断 task/layout 的文件名：
    例如: expert76_traj -> expert76_task.json / expert76_layout.json
    """
    folder_name = os.path.basename(traj_dir)        # expert76_traj
    prefix = folder_name.replace("_traj", "")       # expert76

    task_path = os.path.join(traj_dir, f"{prefix}_task.json")
    layout_path = os.path.join(traj_dir, f"{prefix}_layout.json")

    if not (os.path.exists(task_path) and os.path.exists(layout_path)):
        print(f"[跳过] {traj_dir} 中没有匹配的 task/layout json")
        return None, None, prefix

    with open(task_path, "r", encoding="utf-8") as f:
        task = json.load(f)
    with open(layout_path, "r", encoding="utf-8") as f:
        layout = json.load(f)

    return task, layout, prefix


def plot_board(task: dict, layout: dict, title: str, save_path: str):
    boundary = task["boundary"]
    components = task["components"]
    nets = task.get("nets", [])

    comp_dict = {c["name"]: c for c in components}

    fig, ax = plt.subplots()

    # 1. 画板子外框
    bx = [p[0] for p in boundary] + [boundary[0][0]]
    by = [p[1] for p in boundary] + [boundary[0][1]]
    ax.plot(bx, by, linewidth=1)

    all_x = bx[:]
    all_y = by[:]

    # 2. 画元件和焊盘
    for name, place in layout.items():
        if name not in comp_dict:
            continue

        comp = comp_dict[name]
        w = comp["w"]
        h = comp["h"]
        pads = comp["pads"]

        # 假设 x, y 是元件外框左下角坐标
        x0 = place["x"]
        y0 = place["y"]

        # 元件矩形
        rect = Rectangle((x0, y0), w, h, linewidth=1, fill=False)
        ax.add_patch(rect)

        # 元件名
        ax.text(
            x0 + w / 2,
            y0 + h / 2,
            name,
            ha="center",
            va="center",
            fontsize=8,
        )

        # 焊盘小圆点
        for pad in pads:
            px_local, py_local = pad
            px = x0 + px_local
            py = y0 + py_local

            pad_radius = min(w, h) * 0.04
            circle = Circle((px, py), pad_radius)
            ax.add_patch(circle)

            all_x.append(px)
            all_y.append(py)

        # 更新范围
        all_x.extend([x0, x0 + w])
        all_y.extend([y0, y0 + h])

    # 3. 画 nets（虚线连焊盘）
    if nets:
        for net in nets:
            xs = []
            ys = []
            for comp_name, pad_idx in net:
                if comp_name not in comp_dict or comp_name not in layout:
                    continue

                comp = comp_dict[comp_name]
                place = layout[comp_name]

                x0 = place["x"]
                y0 = place["y"]

                pad_local = comp["pads"][pad_idx]
                px = x0 + pad_local[0]
                py = y0 + pad_local[1]

                xs.append(px)
                ys.append(py)

            if len(xs) >= 2:
                ax.plot(xs, ys, linestyle="--", linewidth=0.8)

    # 4. 坐标轴设置 & 保存图片
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)

    if all_x and all_y:
        margin = 2.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name in os.listdir(ROOT_DIR):
        traj_dir = os.path.join(ROOT_DIR, name)

        # 只处理文件夹，并且名字以 "_traj" 结尾
        if not os.path.isdir(traj_dir):
            continue
        if not name.endswith("_traj"):
            continue

        task, layout, prefix = load_task_and_layout(traj_dir)
        if task is None or layout is None:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{prefix}.png")
        print(f"[处理] {prefix} -> {out_path}")
        plot_board(task, layout, title=prefix, save_path=out_path)

    print("全部处理完成。图片都在:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
