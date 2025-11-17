#把待摆放的元件名字按照类型分组并排出一个执行顺序——先放“chip”，再放“core”，最后放其它（normal
def chip_oriented_sequence(task):
    comps = task.get("components", [])
    def typ(c):  # 统一小写
        return str(c.get("type", "")).lower()
    chips   = [c["name"] for c in comps if typ(c) == "chip"]
    cores   = [c["name"] for c in comps if typ(c) == "core"]
    normals = [c["name"] for c in comps if typ(c) not in ("chip", "core")]
    return chips + cores + normals
