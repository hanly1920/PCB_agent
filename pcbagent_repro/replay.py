#每个堆元素是四元组：(priority, counter, score, payload)
#priority：越大越好——注意这里存正数，这样堆顶元素就是“最差”的（pr 最小）；这与后续用 heappushpop 时“淘汰最差”天然吻合
#counter：全局计数器，保证插入顺序（避免优先级相同时堆无法比较报错）当 pr 相等时，Python 需要继续比较第二关键字；如果直接比较 payload（可能是 numpy/tensor/自定义对象）会报“不可比较”的错，所以引入严格递增的整数作为第二关键字
#score：轨迹得分
#payload：样本本体，轨迹数据本体（可以是任意对象，这里是 (steps, traj) 二元组）
# replay.py  —— 方案A：优先级为正 (pr)，容量满时 heappushpop 淘汰“最差”
import heapq  #Python 的小顶堆实现。
import itertools #用itertools.count()生成全局自增计数器，用于打破平局（当两个样本优先级相同，靠计数器保证堆内元素可比较、顺序稳定）
from typing import List, Tuple, Any #类型注解，用于静态检查和阅读

class PrioritizedReplay:
    """
    一个容量受限的优先回放池：
      - 堆元素格式： (pr, counter, score, payload)
        pr:       优先级（正数，越大越好，通常 pr = score ** alpha）
        counter:  自增整型，打破平局，避免比较 payload（里面可能有 numpy/tensor）
        score:    原始分数（训练时做权重/统计）
        payload:  任意对象（例如 (steps, traj)）
      - 容量满时：heappushpop 会弹出“最小的 pr”（即最差样本），从而保留更好的样本
      - 训练选样：topk(k, with_scores=True) 返回 [(payload, score), ...]
    """
    #alpha: 优先级指数，capacity: 最大容量
    #alpha调节优先级非线性，alpha>1时高分样本更容易被保留和采样，alpha=0时退化为普通回放池,0<alpha<1时拉平差距（更均匀）
    def __init__(self, alpha: float = 1.0, capacity: int = 10000):
        self.alpha = alpha
        self.capacity = int(capacity)
        self.data: List[Tuple[float, int, float, Any]] = []  # 小顶堆：(pr, counter, score, payload)
        self._counter = itertools.count() #生成一个无上限自增迭代器 每插入一个元素就调用 next(self._counter) 获取一个唯一整数，保证严格单调递增

    #payload: 你要存的样本内容，score: 样本评分
    def add(self, payload: Any, score: float) -> None:
        """插入一个样本；容量满则淘汰当前“最差”样本。"""
        s = float(score) #把score转成内置浮点，防止numpy.float32导致比较报错
        pr = (s ** self.alpha) if self.alpha != 0 else 1.0  #计算优先级
        entry = (pr, next(self._counter), s, payload)
        if len(self.data) < self.capacity:
            heapq.heappush(self.data, entry)             #没满容量，直接插入 O(log n)
        else:
            heapq.heappushpop(self.data, entry)          # 满了O(log n)：更差者被弹出

    def __len__(self) -> int:  #返回当前池内样本条数
        return len(self.data)

    # 方便在 if buffer: 场景中判断
    def __bool__(self) -> bool:
        return bool(self.data)

    def topk(self, k: int, with_scores: bool = False):
        """
        取优先级最高的 k 个样本（不破坏堆）。
        返回：
          - with_scores=False: [payload, ...]
          - with_scores=True : [(payload, score), ...]
        复杂度：O(n log k)
        """
        if k <= 0 or not self.data:
            return []
        best = heapq.nlargest(min(k, len(self.data)), self.data)  # 按 pr 从大到小
        #用 heapq.nlargest(m, iterable) 从可迭代对象里取 m 个“最大”的元素，且不会修改 self.data ；min(k, len(self.data))：防止 k 大于现有样本数
        if not with_scores:
            return [e[3] for e in best] #best列表里每个元素e是 (pr, counter, score, payload)
        else:
            return [(e[3], e[2]) for e in best]  # (payload, score)

    def pop(self):
        """
        弹出“最好”的样本（pr 最大）。
        注意：由于 heapq 是小顶堆，这里需要 O(n) 的 remove + O(n) 的 heapify。
        训练通常不需要频繁 pop，推荐用 topk；若需高频弹出最好，可改为双堆结构。
        """
        if not self.data:
            return None, None
        best = heapq.nlargest(1, self.data)[0]   # 在小顶堆中找最大元素 O(n)，这一步不会破坏堆结构
        self.data.remove(best)                   # 从底层列表删除这个元素，按值删除 O(n)
        heapq.heapify(self.data)                 # 删除后列表不再满足堆性质，重建堆，O(n)
        _, _, score, payload = best
        return payload, score

    # 可选：弹出“最差”的样本
    def pop_worst(self):
        """弹出优先级最低的样本（堆顶），复杂度 O(log n）。"""
        if not self.data:
            return None, None
        pr, _, score, payload = heapq.heappop(self.data)
        return payload, score

    # 可选：窥视，查看最优/最差样本但不删除
    def peek_best(self):
        """查看最优样本但不删除。"""
        if not self.data:
            return None, None
        pr, _, score, payload = heapq.nlargest(1, self.data)[0]
        return payload, score

    #查看最差样本但不删除
    def peek_worst(self):
        """查看最差样本但不删除（堆顶）。"""
        if not self.data:
            return None, None
        pr, _, score, payload = self.data[0]
        return payload, score
