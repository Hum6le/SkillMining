"""Download curated Flaticon PNG previews and extend the local Markdown gallery."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
import csv

from PIL import Image
import requests


ROOT = Path(__file__).resolve().parent
GALLERY = ROOT.parent / "FLATICON_ICON_GALLERY.md"
START = "<!-- MORE_ICONS_START -->"
END = "<!-- MORE_ICONS_END -->"

# Category, displayed name, Flaticon page slug, icon ID.
ICONS = [
    ("对话与轨迹", "聊天 A", "chat", 2697802),
    ("对话与轨迹", "聊天 B", "chat", 11401343),
    ("对话与轨迹", "聊天 C", "chat", 17558469),
    ("对话与轨迹", "路径 A", "route", 17959920),
    ("对话与轨迹", "路径 B", "route", 9644584),
    ("对话与轨迹", "路径 C", "route", 1989415),
    ("图结构与分支", "图 A", "graph", 10700751),
    ("图结构与分支", "图 B", "graph", 9387244),
    ("图结构与分支", "图 C", "graph", 8831212),
    ("图结构与分支", "分支 A", "branch", 12112618),
    ("图结构与分支", "分支 B", "branch", 3176735),
    ("图结构与分支", "分支 C", "branch", 13955692),
    ("图结构与分支", "流程图 A", "flowchart", 11924379),
    ("图结构与分支", "流程图 B", "flowchart", 17051384),
    ("图结构与分支", "决策树", "decision-tree", 1809086),
    ("图结构与分支", "重试 A", "retry", 11967571),
    ("图结构与分支", "重试 B", "retry", 18796890),
    ("Skill 与资源", "文档 A", "document", 15536883),
    ("Skill 与资源", "文档 B", "document-file", 9812608),
    ("Skill 与资源", "文档 C", "document", 10040231),
    ("Skill 与资源", "数据库 A", "database", 11313468),
    ("Skill 与资源", "数据库 B", "database", 10014509),
    ("Skill 与资源", "文件夹 A", "folder", 11971043),
    ("Skill 与资源", "文件夹 B", "folder", 2277214),
    ("Skill 与资源", "手册 A", "book", 9425555),
    ("Skill 与资源", "手册 B", "book", 10335255),
    ("Skill 与资源", "图层 A", "layers", 11933987),
    ("Skill 与资源", "图层 B", "layers", 18750915),
    ("检索与选择", "放大镜 A", "magnifying-glass", 11250018),
    ("检索与选择", "放大镜 B", "magnifying-glass", 3096110),
    ("检索与选择", "过滤 A", "filter", 18852122),
    ("检索与选择", "过滤 B", "filter", 15598445),
    ("检索与选择", "目标 A", "target", 18626000),
    ("检索与选择", "目标 B", "target", 5738049),
    ("Agent 与训练", "机器人 A", "robot", 11764098),
    ("Agent 与训练", "机器人 B", "robot", 8185880),
    ("Agent 与训练", "训练 A", "training", 14274003),
    ("Agent 与训练", "训练 B", "training", 14274279),
    ("反馈与更新", "反馈 A", "feedback", 12079993),
    ("反馈与更新", "反馈 B", "feedback", 17276622),
    ("反馈与更新", "更新 A", "update", 18718778),
    ("反馈与更新", "更新 B", "update", 10203256),
    ("反馈与更新", "循环 A", "loop", 5890386),
    ("反馈与更新", "循环 B", "loop", 10257739),
    ("评测与边界", "测试 A", "test", 10690900),
    ("评测与边界", "测试 B", "test", 18812510),
    ("评测与边界", "评估 A", "evaluation", 18710069),
    ("评测与边界", "评估 B", "evaluation", 9926288),
    ("评测与边界", "统计 A", "statistics", 18550268),
    ("评测与边界", "统计 B", "statistics", 9783979),
    ("评测与边界", "防护 A", "shield", 12948732),
    ("评测与边界", "防护 B", "shield", 11191844),
    ("定性案例", "正确 A", "check-mark", 7176960),
    ("定性案例", "正确 B", "check-mark", 9458964),
    ("定性案例", "错误 A", "error-message", 11881183),
    ("定性案例", "错误 B", "error", 11825916),
]


def fetch(item):
    category, name, slug, icon_id = item
    filename = f"{slug}_{icon_id}.png"
    path = ROOT / filename
    if not path.exists():
        url = f"https://cdn-icons-png.flaticon.com/512/{str(icon_id)[:-3]}/{icon_id}.png"
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.flaticon.com/"},
            timeout=30,
        )
        response.raise_for_status()
        if response.headers.get("content-type", "").split(";")[0] != "image/png":
            raise ValueError(f"Unexpected content type for {url}")
        with Image.open(BytesIO(response.content)) as image:
            image.verify()
        path.write_bytes(response.content)
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (512, 512):
            raise ValueError(f"Unexpected image format: {path}")
    return item


def main():
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, item): item for item in ICONS}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append((futures[future], str(exc)))

    results.sort(key=lambda item: ICONS.index(item))
    with (ROOT / "additional_icons.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "name", "icon_id", "local_png", "source_page"])
        for category, name, slug, icon_id in results:
            writer.writerow([category, name, icon_id, f"{slug}_{icon_id}.png",
                             f"https://www.flaticon.com/free-icon/{slug}_{icon_id}"])

    parts = ["## 更多候选图标", "", f"本批新增 **{len(results)}** 个 512 × 512 PNG 预览。图标保留 Flaticon 原始配色；点击缩略图打开本地文件，点击名称打开来源页面。", ""]
    for category in dict.fromkeys(item[0] for item in results):
        parts.extend([f"### {category}", "", "| 预览 | 名称与来源 | 预览 | 名称与来源 |", "|---|---|---|---|"])
        group = [item for item in results if item[0] == category]
        cells = []
        for _, name, slug, icon_id in group:
            filename = f"{slug}_{icon_id}.png"
            link = f"https://www.flaticon.com/free-icon/{slug}_{icon_id}"
            cells.append((f'[<img src="flaticon_icons/{filename}" width="88" alt="{name}">](flaticon_icons/{filename})',
                          f"[{name}]({link})<br>#{icon_id}"))
        for index in range(0, len(cells), 2):
            left = cells[index]
            right = cells[index + 1] if index + 1 < len(cells) else ("", "")
            parts.append("| " + " | ".join((*left, *right)) + " |")
        parts.append("")
    parts.append("新增图标的来源链接也保存在 [additional_icons.csv](flaticon_icons/additional_icons.csv)。选定实际使用的图标后，请到对应页面核对作者并署名。")
    block = START + "\n" + "\n".join(parts) + "\n" + END

    original = GALLERY.read_text(encoding="utf-8")
    if START in original and END in original:
        original = original[:original.index(START)] + block + original[original.index(END) + len(END):]
    else:
        original = original.rstrip() + "\n\n" + block + "\n"
    GALLERY.write_text(original, encoding="utf-8")
    print(f"Downloaded and indexed {len(results)} icons; failures: {len(errors)}")
    for item, error in errors:
        print(f"FAILED {item}: {error}")


if __name__ == "__main__":
    main()
