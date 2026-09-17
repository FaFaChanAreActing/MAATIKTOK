"""抖音自动续火花 agent —— Custom 层（单遍扫描 + 全量健壮性）

职责划分（对应需求文档 §4 / §5 / §6 / §8）：
  - 整屏扫描带半火花图标的好友（色块连通 + 颜色判定，非模板匹配）、昵称归一化
  - 当日已发集合读写：user_data/sent_YYYYMMDD.json（含 sent / failed / updated_at）
  - 「今日是否已发」判断：本地集合优先，其次聊天页 OCR 时间标签 + 气泡归属
  - 续火文案随机：config/greetings.json
  - 发送成功校验 + 失败重试 1 次 + failed 落盘 + 发送后随机间隔
  - 到顶判定：滑动前后截图无变化，或连续 N 屏无新增半火花好友
  - 弹窗点掉、点错好友回退

pipeline 侧只负责无状态编排：开 App / 进好友页 / 上滑 / 返回 / 关 App / 分支。
运行分辨率 1280x720。
"""
import json
import os
import random
import re
import time
from datetime import date, datetime

import numpy

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition

# ---------------- 路径 ----------------
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_AGENT_DIR)
USER_DATA_DIR = os.path.join(_ROOT_DIR, "user_data")
DEBUG_DIR = os.path.join(USER_DATA_DIR, "debug")   # 校准期整屏截图落盘目录
GREETING_FILE = os.path.join(_ROOT_DIR, "config", "greetings.json")

# ---------------- 坐标（1280x720；各 ROI 的实际生效值在 pipeline 辅助节点里）----------------
COMP_ROI = (0, 70, 1280, 610)          # 上滑端点检测的截图对比区域
INPUT_BOX = (640, 680)                 # 聊天页输入框
SEND_BTN = (1200, 680)                 # 发送按钮
OWN_BUBBLE_MIN_CX = 700                # 我方气泡中心 x 下限（靠右对齐判定，doc §5.3）

# 火花图标颜色判定（doc §5.1 的 TemplateMatch 方案替换为色块连通检测，见 _icon_state）
NICK_SCAN_X0 = 95                      # 昵称列扫描左边界（头像右缘 ≈85，避免把头像照片算进来）
NICK_SCAN_X1 = 520                     # 昵称列扫描右边界（时间戳在 x>1170，天然排除）
ICON_RUN_MIN = 5                       # 单行内同色连续像素下限：实心图标 ≫5，文字抗锯齿只有 1~2px
ICON_MIN_ROWS = 5                      # 有色列至少要在这么多行里出现长连续段（汉字横笔画只有上下 2 行抗锯齿灰边）
ICON_MIN_W = 7                         # 色块最小宽度（px）
ICON_MAX_W = 45                        # 色块最大宽度（px）
ICON_MIN_PX = 30                       # 色块有效像素下限
ORANGE_RATIO = 1.2                     # 橙像素/灰像素 超过该比值 → 满火花（今天已续）

# ---------------- 可调阈值 ----------------
DIFF_THRESHOLD = 4.0                   # 截图无变化判定（0-255 均值绝对差）
MAX_BOUND_CALLS = 150                  # 端点检测最多尝试次数，超过强制结束（纯安全网，正常到不了）
MAX_SENDS_PER_RUN = 50                 # 单轮发送总条数上限（doc §8 防风控）
SEND_WAIT = 1.5                        # 发送后等待校验（秒）
SEND_RETRY = 1                         # 发送失败重试次数
SLEEP_RANGE = (1.0, 3.0)               # 每次发送后的随机间隔（doc §8）
BACK_WAIT = 1.8                        # 每次返回键后等待时长（等键盘收起/页面切换动画）
BACK_MAX_TRIES = 4                     # 返回键最多按几次（软键盘收起 1 次 + 退出聊天页 1 次，再留余量）
LIST_MIN_ROWS = 3                      # 判定「在消息列表页」所需的最少昵称行数
ROW_PITCH_MIN = 60                     # 列表行距下限（1280x720 实测 ≈83.5）
ROW_PITCH_MAX = 100                    # 列表行距上限

# 昵称行识别（1280x720，实测布局：昵称左边界 x≈96、行高 25；预览行同在 x≈96 但是灰字）
NICK_X_MIN = 85                        # 昵称文本左边界下限
NICK_X_MAX = 115                       # 昵称文本左边界上限
NICK_HEIGHT_MAX = 35                   # 昵称行高度上限（排除 OCR 合并多行的大框）
NICK_DARK_RATIO = 0.02                 # 暗像素占比下限：昵称是黑字，预览/时间戳是灰字（用于剔除预览行）
PREVIEW_GAP_MIN = 18                   # 昵称与紧邻其下方预览行的 y 间距下限（实测 ≈28）
PREVIEW_GAP_MAX = 42                   # 上限（列表行距 ≈85，不会误伤真正的下一行）

# 昵称清洗：去掉可能混入的状态文字
_STATUS_RE = re.compile(r"重燃中|重连中|领火星|续火花|点亮中|已续火|\d+\s*/\s*\d+|^\d+$")
# 昵称尾部天数（"颜亦204" → "颜亦"，避免同一个人被 OCR 读成 204/205 两个名字）
_TAIL_DAYS_RE = re.compile(r"[\s\d]+$")
# 聊天页时间标签：非今日（昨天/前天/月日/星期）
OTHER_DAY_RE = re.compile(r"昨天|前天|\d{1,2}月\d{1,2}日|星期[一二三四五六日]")
# 今日标签：今天 / 刚刚 / 裸时间 HH:MM（聊天页刚发完消息上方显示的就是"刚刚"）
TODAY_RE = re.compile(r"今天|刚刚|^\s*\d{1,2}\s*[:：]\s*\d{2}\s*$")
TIME_LABEL_MAX_LEN = 12                # 时间标签文本长度上限，避免把聊天内容当标签

# 弹窗上可点的关闭类文案（doc §8）
POPUP_KEYWORDS = ("以后再说", "跳过", "我知道了", "稍后再说", "不再提示", "暂不更新", "暂不开启")
# 聊天页底部输入框占位文案（实测 1280x720 为「发消息或按住说话…」）
CHAT_INPUT_HINTS = ("发消息", "按住说话", "说点什么")

# 默认续火文案（config/greetings.json 缺失时使用）
DEFAULT_GREETINGS = ["hi", "在吗", "续一下～", "早"]

# 调试：打印前 N 屏的昵称行 + 暗占比 + 图标判定，用于校准阈值；校准完可置 False
DEBUG_LOG = True
DEBUG_SCREENS = 3

# pipeline 中供 run_recognition 调用的辅助节点名
LIST_OCR_NODE = "模板_列表昵称OCR"
CHAT_OCR_NODE = "模板_聊天页OCR"
INPUT_OCR_NODE = "模板_输入框OCR"
TITLE_OCR_NODE = "模板_聊天页标题OCR"
POPUP_OCR_NODE = "模板_弹窗OCR"

# ---------------- 运行期状态（init_state 重置）----------------
_state = {
    "processed": set(),      # 本次运行已点击处理过（发送成功/今日已发/点错），不再重复点击
    "sent_today": set(),     # 磁盘记录的今日已发昵称
    "last_matched": "",      # 当前处理中的好友昵称
    "no_new_screens": 0,     # 连续无新增半火花好友的屏数
    "prev_shot": None,       # check_bottom 上一次截图（裁剪后）
    "bound_calls": 0,
    "swipes": 0,             # 已上滑次数
    "sends": 0,              # 本次运行成功发送条数
    "dbg_screens": 0,        # 已打印调试信息的屏数
}


# ================= 持久化 / 工具 =================

def _today():
    return date.today().isoformat()


def _today_file():
    return os.path.join(USER_DATA_DIR, "sent_{}.json".format(_today().replace("-", "")))


def _load_today():
    """读当天文件；不存在或跨天则返回空结构（doc §6.1）"""
    empty = {"date": _today(), "sent": [], "failed": [], "updated_at": ""}
    try:
        with open(_today_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return empty
    if not isinstance(data, dict) or data.get("date") != _today():
        return empty
    for key in ("sent", "failed"):
        if not isinstance(data.get(key), list):
            data[key] = []
    return data


def _save_today(data):
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    data["date"] = _today()
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_today_file(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _record_sent(nick):
    """成功发送：立即追加落盘（doc §6.1）"""
    data = _load_today()
    if not any(_nick_match(nick, n) for n in data["sent"]):
        data["sent"].append(nick)
        _save_today(data)
    _state["sent_today"].add(nick)
    _state["processed"].add(nick)
    _state["sends"] += 1


def _record_failed(nick, reason):
    """发送失败：记入 failed，不写入 sent（doc §8）"""
    data = _load_today()
    if not any(_nick_match(nick, n) for n in data["failed"]):
        data["failed"].append(nick)
        _save_today(data)
    _state["processed"].add(nick)
    print("[failed] {} 发送失败：{}".format(nick, reason))


def _load_greetings():
    """读 config/greetings.json，缺失则创建默认配置（doc §6.2）"""
    try:
        with open(GREETING_FILE, "r", encoding="utf-8") as f:
            texts = json.load(f)
        texts = [str(t) for t in texts if str(t).strip()]
        if texts:
            return texts
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        pass
    try:
        os.makedirs(os.path.dirname(GREETING_FILE), exist_ok=True)
        with open(GREETING_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_GREETINGS, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print("[greetings] 写入默认配置失败：{}".format(e))
    return list(DEFAULT_GREETINGS)


def _clean_nickname(text):
    """清洗 OCR 文本得到稳定昵称：去掉混入的状态文字与尾部天数"""
    t = re.sub(_STATUS_RE, "", text or "").strip()
    return _TAIL_DAYS_RE.sub("", t).strip()


def _has_tail_days(text):
    """昵称行尾部是否带连续天数。

    火花好友的行一定是「昵称 + 图标 + 天数」（"颜亦205" / "鸿声渡晚3"），
    聊天页里分享卡片、底部快捷表情栏的文字都没有天数，靠这一条可以直接排除。
    """
    return bool(_TAIL_DAYS_RE.search(text or ""))


def _imgs_same(a, b):
    if a is None or b is None:
        return False
    try:
        if a.shape != b.shape:
            return False
        step = 10
        da = a[::step, ::step].astype(numpy.int16)
        db = b[::step, ::step].astype(numpy.int16)
        return float(numpy.mean(numpy.abs(da - db))) < DIFF_THRESHOLD
    except Exception:
        return False


def _nick_match(a, b):
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 2 and a in b:
        return True
    if len(b) >= 2 and b in a:
        return True
    return False


def _in_set(nick, s):
    return any(_nick_match(nick, x) for x in s)


# ================= OCR 辅助 =================

def _ocr(context, node, image):
    """跑一个 OCR 节点，返回 [(text, box), ...]"""
    reco = context.run_recognition(node, image)
    if reco is None:
        return []
    out = []
    for r in getattr(reco, "all_results", []) or []:
        text = str(getattr(r, "text", "") or "").strip()
        box = getattr(r, "box", None)
        if text and box is not None:
            out.append((text, list(box)))
    return out


def _lum_sat(sub):
    """BGR(numpy) → (亮度, 饱和度)，亮度用整数近似 BT.601

    必须用 int32：r*299 在 int16 下会溢出（255*299=76245），导致亮度算成负数。
    """
    b = sub[..., 0].astype(numpy.int32)
    g = sub[..., 1].astype(numpy.int32)
    r = sub[..., 2].astype(numpy.int32)
    mx = numpy.maximum(numpy.maximum(r, g), b)
    mn = numpy.minimum(numpy.minimum(r, g), b)
    lum = (r * 299 + g * 587 + b * 114) // 1000
    return lum, (mx - mn)


def _dark_ratio(image, box):
    """box 内暗像素占比。昵称是黑字（≈0.1）；预览行/时间戳是灰字（<0.01）"""
    x, y, w, h = box
    sub = image[max(0, y):y + h, max(0, x):x + w]
    if sub.size == 0:
        return 0.0
    lum, _ = _lum_sat(sub)
    return float((lum < 100).mean())


def _run_groups(mask):
    """把「多行都有水平连续同色像素 ≥ ICON_RUN_MIN」的列标出来，再合并成列组

    实心图标每一行都有 10+ 连续同色像素，因此它的列会被反复记到；
    文字抗锯齿只有 1~2px 的灰边（够不到 ICON_RUN_MIN），汉字横笔画最多带来
    上下 2 行通长灰边（够不到 ICON_MIN_ROWS），分隔线这类单行通栏灰线同理被滤掉。
    """
    h, w = mask.shape
    hits = numpy.zeros(w, dtype=numpy.int16)
    for yy in range(h):
        row = mask[yy]
        run = 0
        for xx in range(w + 1):
            if xx < w and row[xx]:
                run += 1
                continue
            if run >= ICON_RUN_MIN:
                hits[xx - run:xx] += 1
            run = 0
    cols = hits >= ICON_MIN_ROWS
    groups, start = [], None
    for xx in range(w):
        if cols[xx] and start is None:
            start = xx
        elif not cols[xx] and start is not None:
            groups.append((start, xx))
            start = None
    if start is not None:
        groups.append((start, w))
    return groups


def _icon_state(image, box):
    """判断昵称行里的火花图标：'orange'（满火花/今天已续）/ 'gray'（半火花/待续）/ None（无火花）

    抖音消息列表布局为「昵称 + [火花图标] + 连续天数」，图标是实心色块。
    取最右侧的合格色块：昵称自带彩色 emoji 时（如「🔥只因」），火花图标总在天数左侧、更靠右。
    """
    x, y, w, h = box
    y0, y1 = max(0, y + 3), min(image.shape[0], y + h - 3)
    sub = image[y0:y1, NICK_SCAN_X0:NICK_SCAN_X1]
    if sub.size == 0:
        return None
    lum, sat = _lum_sat(sub)
    b = sub[..., 0].astype(numpy.int16)
    g = sub[..., 1].astype(numpy.int16)
    r = sub[..., 2].astype(numpy.int16)

    orange = (r > 140) & ((r - b) > 50) & ((r - g) > 25)
    grayish = (sat < 30) & (lum > 105) & (lum < 228)
    mask = orange | grayish

    for gx0, gx1 in reversed(_run_groups(mask)):
        if not (ICON_MIN_W <= gx1 - gx0 <= ICON_MAX_W):
            continue
        seg = mask[:, gx0:gx1]
        if int(seg.sum()) < ICON_MIN_PX:
            continue
        o = int(orange[:, gx0:gx1].sum())
        gr = int(grayish[:, gx0:gx1].sum())
        return "orange" if o > gr * ORANGE_RATIO else "gray"
    return None


def _dump_screen(image, tag):
    """把整屏截图落盘，用于对照 OCR 坐标校准布局（仅调试期）

    argv.image 是 BGR，PIL 要 RGB，所以按 [::-1] 反一下通道。
    """
    try:
        from PIL import Image
        os.makedirs(DEBUG_DIR, exist_ok=True)
        path = os.path.join(DEBUG_DIR, "screen_{}_{}.png".format(
            _today().replace("-", ""), tag))
        Image.fromarray(image[..., ::-1]).save(path)
        print("[debug] 整屏截图已保存 {}".format(path))
    except Exception as e:
        print("[debug] 保存整屏截图失败：{}".format(e))


def _debug_screen(image, ocr):
    """打印每屏昵称列上的所有行 + 暗占比 + 图标判定 + 是否带天数，用于校准阈值"""
    if not DEBUG_LOG or _state["dbg_screens"] >= DEBUG_SCREENS:
        return
    _state["dbg_screens"] += 1
    _dump_screen(image, _state["dbg_screens"])
    rows = _find_nick_rows(image, ocr)
    print("[debug] ======== 第 {} 屏（昵称列 {} 行）========".format(
        _state["dbg_screens"], len(rows)))
    for text, box in sorted(rows, key=lambda x: x[1][1]):
        # _icon_state 可能返回 None，必须先转成字符串再格式化
        icon = _icon_state(image, box) or "-"
        print("[debug]   y={:>4} box={} 文本={!r:<26} 暗占比={:.3f} 图标={:<7} {}".format(
            box[1], box, text, _dark_ratio(image, box), icon,
            "有天数" if _has_tail_days(text) else "无天数→排除"))
    others = [t for t, b in ocr if not (NICK_X_MIN <= b[0] <= NICK_X_MAX)]
    print("[debug]   被过滤的非昵称列 OCR {} 条: {}".format(len(others), others[:12]))


def _find_nick_rows(image, ocr):
    """找出昵称行：左对齐昵称列（x≈96）+ 文字为黑色

    实测布局（1280x720）：昵称左边界 x≈96、行高 25；
    昵称下方 30px 处还有一行预览文字，同样从 x≈96 开始但是灰字，
    所以用暗像素占比区分（昵称 ≈0.05~0.20，预览 ≈0.00）。

    返回 [(nick_text, nick_box), ...]
    """
    rows = []
    for text, box in ocr:
        if not (NICK_X_MIN <= box[0] <= NICK_X_MAX):
            continue
        if box[3] > NICK_HEIGHT_MAX:
            continue
        if _dark_ratio(image, box) < NICK_DARK_RATIO:
            continue
        rows.append((text, box))

    # 剔除预览行：昵称行与它下方 28~30px 处的预览行只差 28px，而列表行距是 85px。
    # 预览行大多是灰字会被暗占比滤掉，但带 emoji 的预览（"😂😂😂"）是彩色的，
    # 既过得了暗占比、又会被 _icon_state 判成 orange，只能靠这个间距剔掉。
    ys = sorted(b[1] for _, b in rows)
    preview_ys = set()
    for i in range(1, len(ys)):
        if PREVIEW_GAP_MIN <= ys[i] - ys[i - 1] <= PREVIEW_GAP_MAX:
            preview_ys.add(ys[i])
    return [(t, b) for t, b in rows if b[1] not in preview_ys]


def _is_today_label(text):
    """判断 OCR 文本是否为「今天」的时间标签（doc §5.3）"""
    t = (text or "").strip()
    if not t or len(t) > TIME_LABEL_MAX_LEN:
        return False
    if OTHER_DAY_RE.search(t):
        return False
    return bool(TODAY_RE.search(t))


def _chat_sent_today(context, image):
    """聊天页判断今日是否已发：时间标签 + 气泡归属（doc §5.3）

    规则：找到最靠上的「今天/HH:MM」标签，看其下方是否存在靠右（我方）气泡。
    任何不确定情况一律按「未发」处理，靠当日已发集合防止重复发。
    """
    ocr = _ocr(context, CHAT_OCR_NODE, image)
    labels = [(t, b) for t, b in ocr if _is_today_label(t)]
    if not labels:
        return False, "聊天页无今日时间标签"
    first_y = min(b[1] for _, b in labels)
    for text, box in ocr:
        if _is_today_label(text):
            continue
        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        if cy >= first_y and cx >= OWN_BUBBLE_MIN_CX:
            return True, "今日标签下方存在我方气泡 {!r}".format(text)
    return False, "今日标签下方无我方气泡"


def _on_list_page(context, image, ocr=None):
    """正向判断「当前在消息列表页」

    判据：昵称列（x≈96）上有 ≥LIST_MIN_ROWS 行左对齐黑色文字，且相邻行距落在
    列表的行距范围内（1280x720 实测 ≈83.5px，日志中 7 行实测 82~87）。

    不看「输入框占位文案」这类反向特征——聊天页真实占位是「发消息或按住说话…」，
    关键词根本对不上；也不只看「有没有文字」——聊天页的分享卡片、底部快捷表情栏
    都可能出现黑色文字，但凑不出 3 行等间距。
    """
    if ocr is None:
        ocr = _ocr(context, LIST_OCR_NODE, image)
    rows = sorted(b[1] for _, b in _find_nick_rows(image, ocr))
    if len(rows) < LIST_MIN_ROWS:
        return False
    gaps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
    aligned = sum(ROW_PITCH_MIN <= gap <= ROW_PITCH_MAX for gap in gaps)
    return aligned >= 2


def _in_chat_page(context, image):
    """当前是否是聊天页：底部输入框占位文案命中（实测「发消息或按住说话…」）

    只把「命中」当证据，不命中不下结论 —— 是否回到列表由 _on_list_page 正向判断。
    两个判据一起用：聊天页里分享卡片的文字可能凑够 _on_list_page 的行数与行距，
    但只要输入框占位文案还在，就说明人还在聊天页。
    """
    for text, _ in _ocr(context, INPUT_OCR_NODE, image):
        if any(hint in text for hint in CHAT_INPUT_HINTS):
            return True
    return False


# ================= 自定义动作 =================

@AgentServer.custom_action("init_state")
class InitStateAction(CustomAction):
    """任务开始：重置运行态并载入当天已发集合"""

    def run(self, context: Context, argv: CustomAction.RunArg):
        data = _load_today()
        _state.update(
            processed=set(),
            sent_today={n for n in data["sent"] if n},
            last_matched="",
            no_new_screens=0,
            prev_shot=None,
            bound_calls=0,
            swipes=0,
            sends=0,
            dbg_screens=0,
        )
        print("[init_state] 日期={} 今日已发={} 条: {}".format(
            data["date"], len(_state["sent_today"]), sorted(_state["sent_today"])))
        return True


@AgentServer.custom_action("mark_processed")
class MarkProcessedAction(CustomAction):
    """已发/点错分支：只加入运行态 processed，不写盘（doc §6.1 只记真正发送的）"""

    def run(self, context: Context, argv: CustomAction.RunArg):
        nick = _state.get("last_matched", "")
        if nick:
            _state["processed"].add(nick)
            print("[mark_processed] {} 标记已处理（跳过发送）".format(nick))
        return True


@AgentServer.custom_action("send_greeting")
class SendGreetingAction(CustomAction):
    """发送续火文案 + 成功校验 + 重试 + 落盘（doc §5.4 / §8）"""

    def run(self, context: Context, argv: CustomAction.RunArg):
        nick = _state.get("last_matched", "")
        if not nick:
            print("[send_greeting] 无当前好友昵称，跳过")
            return True
        text = random.choice(_load_greetings())
        ctrl = context.tasker.controller

        for attempt in range(SEND_RETRY + 1):
            ctrl.post_click(INPUT_BOX[0], INPUT_BOX[1]).wait()
            time.sleep(0.8)
            ctrl.post_input_text(text).wait()
            time.sleep(0.6)
            ctrl.post_click(SEND_BTN[0], SEND_BTN[1]).wait()
            time.sleep(SEND_WAIT)
            image = ctrl.post_screencap().wait().get()
            ok, why = _verify_sent(context, image, text)
            if ok:
                _record_sent(nick)
                print("[send_greeting] {} 发送成功（{!r}，{}，第 {} 次尝试）".format(
                    nick, text, why, attempt + 1))
                time.sleep(random.uniform(*SLEEP_RANGE))  # 风控间隔
                return True
            print("[send_greeting] {} 第 {} 次发送未确认：{}".format(nick, attempt + 1, why))

        _record_failed(nick, "重试 {} 次仍未确认发送成功（文案 {!r}）".format(SEND_RETRY, text))
        return True


def _verify_sent(context, image, text):
    """发送成功判定：消息出现在聊天流底部 或 输入框已清空（doc §5.4）"""
    for t, _ in _ocr(context, CHAT_OCR_NODE, image):
        if t.strip() == text:
            return True, "消息已出现在聊天流"
    still = [t for t, _ in _ocr(context, INPUT_OCR_NODE, image) if t.strip() == text]
    if not still:
        return True, "输入框已清空"
    return False, "输入框仍残留文案，疑似未发送"


@AgentServer.custom_action("back_to_list")
class BackToListAction(CustomAction):
    """返回消息列表：最多按 BACK_MAX_TRIES 次返回键，每次都正向确认（doc §5.4）

    发完消息后软键盘通常还开着，第一次返回只关掉键盘，所以必须重试到
    「真的能识别到昵称列」为止，否则后续会把聊天页当成列表页误判到顶。
    """

    def run(self, context: Context, argv: CustomAction.RunArg):
        ctrl = context.tasker.controller
        back_count = 0
        for i in range(BACK_MAX_TRIES + 1):
            image = ctrl.post_screencap().wait().get()
            if _on_list_page(context, image) and not _in_chat_page(context, image):
                print("[back_to_list] 已回到消息列表（按返回键 {} 次）".format(back_count))
                _state["last_matched"] = ""
                return True
            if i == BACK_MAX_TRIES:
                break
            ctrl.post_click_key(4).wait()
            time.sleep(BACK_WAIT)
            back_count += 1
        print("[back_to_list] 警告：按了 {} 次返回键仍未确认回到消息列表，本轮可能提前结束".format(back_count))
        _state["last_matched"] = ""
        return True


# ================= 自定义识别 =================

@AgentServer.custom_recognition("match_unsent_spark")
class MatchUnsentSparkRecognition(CustomRecognition):
    """当前屏带半火花（灰色图标）、且本次运行未处理过的好友（doc §4 / §5.1）

    命中返回昵称 box 供 Click；全屏无可处理好友返回 None，
    pipeline 落到兄弟节点「处理无匹配」去检测端点/上滑。
    """

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
        if _state["sends"] >= MAX_SENDS_PER_RUN:
            print("[match_unsent_spark] 已达单轮上限 {} 条，停止发送".format(MAX_SENDS_PER_RUN))
            _state["no_new_screens"] += 1
            return CustomRecognition.AnalyzeResult(box=None, detail={"limit": True})

        image = argv.image
        ocr = _ocr(context, LIST_OCR_NODE, image)
        try:
            _debug_screen(image, ocr)   # 调试输出不允许影响主流程
        except Exception as e:
            print("[debug] 打印调试信息失败，已忽略：{}".format(e))

        # 页面守卫：还在聊天页时绝不匹配，否则会把聊天页的卡片/表情栏文字当成好友
        if _in_chat_page(context, image):
            print("[match_unsent_spark] 当前仍在聊天页，跳过本屏匹配（不计入无新增）")
            return CustomRecognition.AnalyzeResult(box=None, detail={"hit": False, "in_chat": True})

        all_rows = _find_nick_rows(image, ocr)
        if not all_rows:
            # 昵称列一行都没有 → 当前位置不是消息列表页，属于页面异常。
            # 这里绝对不累加 no_new_screens：否则一次翻页失败就会被当成「列表已到底」
            # 而提前收工（实测第 3 屏起连续几屏读不到行，直接把整轮判到顶关掉了 App）。
            print("[match_unsent_spark] 本屏昵称列无任何行（疑似不在列表页），不计入无新增")
            return CustomRecognition.AnalyzeResult(box=None, detail={"hit": False, "no_rows": True})

        # 只有「昵称 + 图标 + 天数」这种完整行才算火花好友行；
        # 昵称下面 30px 处的预览行、聊天页的卡片文字都带不出天数
        nick_rows = [(t, b) for t, b in all_rows if _has_tail_days(t)]

        for nick_text, nick_box in nick_rows:
            nick = _clean_nickname(nick_text)
            # 本次运行已处理过，或当日已发集合已记录 → 直接跳过，不再进聊天页
            if not nick or _in_set(nick, _state["processed"]) or _in_set(nick, _state["sent_today"]):
                continue
            state = _icon_state(image, nick_box)
            if state is None:              # 没有火花图标（如"一个光"），不是火花好友
                continue
            if state == "orange":          # 橙=满火花=今天已续，无需再发
                _state["processed"].add(nick)
                print("[match_unsent_spark] {} 火花已满（今日已续），跳过".format(nick))
                continue
            # 命中即登记为已处理：万一后面点错好友走了「未知页返回」，
            # 也不会在下一屏重复扫到同一个昵称，导致死循环
            _state["processed"].add(nick)
            _state["last_matched"] = nick
            _state["no_new_screens"] = 0
            print("[match_unsent_spark] 命中半火花好友 {}（OCR {!r}，图标 gray）".format(nick, nick_text))
            return CustomRecognition.AnalyzeResult(
                box=nick_box, detail={"nick": nick, "ocr_text": nick_text, "icon": "gray"})

        _state["no_new_screens"] += 1
        print("[match_unsent_spark] 本屏 {} 行昵称、其中 {} 行带天数，均无需处理（连续无新增 {} 屏）".format(
            len(all_rows), len(nick_rows), _state["no_new_screens"]))
        return CustomRecognition.AnalyzeResult(box=None, detail={"hit": False})


@AgentServer.custom_recognition("verify_chat_title")
class VerifyChatTitleRecognition(CustomRecognition):
    """点错好友保护：聊天页标题与目标昵称比对（doc §8）

    命中=标题匹配或无法判断（放行）；未命中=确认点错，返回列表。
    """

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
        nick = _state.get("last_matched", "")
        titles = [t for t, _ in _ocr(context, TITLE_OCR_NODE, argv.image)]
        if not titles:
            print("[verify_chat_title] 标题无法识别，按放行处理")
            return CustomRecognition.AnalyzeResult(box=(0, 0, 10, 10), detail={"unknown": True})
        if any(_nick_match(nick, t) for t in titles):
            print("[verify_chat_title] 标题匹配 {}：{}".format(nick, titles))
            return CustomRecognition.AnalyzeResult(box=(0, 0, 10, 10), detail={"title": titles})
        print("[verify_chat_title] 标题不匹配 {}（实际 {}），判定点错好友".format(nick, titles))
        return CustomRecognition.AnalyzeResult(box=None, detail={"title": titles})


@AgentServer.custom_recognition("check_sent_today")
class CheckSentTodayRecognition(CustomRecognition):
    """今日是否已发：本地集合优先，其次聊天页 OCR（doc §5.3）

    命中=今日已发 → 走「标记已处理」；未命中=未发 → 落到兄弟节点「发送续火文案」。
    """

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
        nick = _state.get("last_matched", "")
        if _in_set(nick, _state["sent_today"]):
            print("[check_sent_today] {} 当日集合已记录，跳过".format(nick))
            return CustomRecognition.AnalyzeResult(box=(0, 0, 10, 10), detail={"sent": True, "by": "local"})
        sent, why = _chat_sent_today(context, argv.image)
        print("[check_sent_today] {} {}：{}".format(nick, "今日已发" if sent else "今日未发", why))
        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 10, 10) if sent else None,
            detail={"sent": sent, "by": "chat_ocr", "reason": why},
        )


@AgentServer.custom_recognition("check_bottom")
class CheckBottomRecognition(CustomRecognition):
    """到顶判定（doc §5.5，但「连续 N 屏无新增」不参与触发）

    命中=到顶：滑动前后截图无变化（且确实在消息列表页），或超过尝试上限。
    未命中=继续上滑。首次调用仅建立基线。
    """

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
        x, y, w, h = COMP_ROI
        cur = argv.image[y:y + h, x:x + w]
        prev = _state.get("prev_shot")
        _state["bound_calls"] += 1
        if prev is None:
            _state["prev_shot"] = cur
            return CustomRecognition.AnalyzeResult(box=None, detail={"end": False, "baseline": True})

        # 「画面没变化」只有在消息列表页才成立：卡在聊天页时画面同样不变，
        # 但那不是到底部，不能就此收工
        same = _imgs_same(cur, prev) and _on_list_page(context, argv.image)
        _state["prev_shot"] = cur
        force = _state["bound_calls"] > MAX_BOUND_CALLS
        if same or force:
            # 「连续无新增」只作诊断输出，不参与判定：列表尾部常常连续很多屏
            # 都是群聊/无火花好友，用它收工会出现「还在滑动就判到顶」的误判。
            print("[check_bottom] 判定到顶（列表页画面未变化={} 超限={}；连续无新增 {} 屏仅供参考）".format(
                same, force, _state["no_new_screens"]))
            return CustomRecognition.AnalyzeResult(
                box=(0, 0, 10, 10), detail={"end": True, "same": same, "force": force})
        _state["swipes"] += 1  # 未到顶，接下来会执行上滑
        return CustomRecognition.AnalyzeResult(box=None, detail={"end": False})


@AgentServer.custom_recognition("handle_popup")
class HandlePopupRecognition(CustomRecognition):
    """弹窗点掉：OCR 命中关闭类文案则返回其位置（doc §8）

    只在开 App 后检查，避免把列表里的消息预览误判成弹窗按钮。
    """

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
        image = argv.image
        for text, box in _ocr(context, POPUP_OCR_NODE, image):
            for kw in POPUP_KEYWORDS:
                if kw in text:
                    print("[handle_popup] 命中弹窗按钮 {!r} {}".format(text, box))
                    return CustomRecognition.AnalyzeResult(box=box, detail={"keyword": kw})
        return CustomRecognition.AnalyzeResult(box=None, detail={"hit": False})