import os
import math

# 颜色配置（Manjaro/Xterm 终端标准 ANSI 字符）
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
GREEN = "\033[1;32m"
CYAN = "\033[1;36m"
RESET = "\033[0m"
BOLD = "\033[1m"


def calc_metrics(file_path):
    """单文件指标核心计算"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        total_len = len(data)
        if total_len == 0:
            return None

        # 1. 统计频次
        counts = [0] * 256
        zeros = 0
        for b in data:
            counts[b] += 1
            if b == 0:
                zeros += 1

        # 2. 计算香农信息熵
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total_len
                entropy -= p * math.log2(p)

        zero_ratio = zeros / total_len
        return entropy, zero_ratio, total_len / (1024 * 1024)
    except Exception:
        return None


def make_bar(ratio, length=20, char="█"):
    """生成漂亮的进度条"""
    filled = int(ratio * length)
    return char * filled + "░" * (length - filled)


def run_scan(target_dir):
    print(f"\n{BOLD}{CYAN}🔮 [Manjaro 字节大模型数据源筛查仪表盘]{RESET}")
    print(f"{CYAN}目标目录: {target_dir}{RESET}")
    print("-" * 85)
    print(f"{BOLD}{'File Name':<28} | {'Size':<7} | {'Information Entropy (0-8)':<22} | {'0x00 Ratio':<15}{RESET}")
    print("-" * 85)

    # 搜集所有文件
    files = []
    for root, _, fs in os.walk(target_dir):
        for f in fs:
            if not f.startswith('.'):
                files.append(os.path.join(root, f))
    files.sort()

    clean_count = 0
    danger_count = 0

    for fp in files:
        metrics = calc_metrics(fp)
        if not metrics:
            continue

        entropy, zero_ratio, size_mb = metrics
        name = os.path.basename(fp)
        if len(name) > 26:
            name = name[:23] + "..."

        # 转换成进度条表现形式
        ent_ratio = entropy / 8.0  # 映射到 0~1 绘制
        ent_bar = make_bar(ent_ratio, length=15)
        zero_bar = make_bar(zero_ratio, length=10)

        # 判定状态与危险分级
        status_color = GREEN
        tag = ""

        if entropy > 7.85:
            status_color = RED
            tag = " ☢️  [加壳/强加密地雷]"
            danger_count += 1
        elif zero_ratio > 0.40:
            status_color = YELLOW
            tag = " ⚠️  [全零空洞黑洞]"
            danger_count += 1
        else:
            clean_count += 1

        # 打印单行精美数据
        print(f"{status_color}{name:<28}{RESET} | "
              f"{size_mb:>5.1f}M | "
              f"{status_color}{entropy:>5.2f}{RESET} {ent_bar} | "
              f"{status_color}{zero_ratio * 100:>5.1f}%{RESET} {zero_bar}{status_color}{tag}{RESET}")

    # 总结报告
    print("-" * 85)
    print(
        f"{BOLD}📊 扫描总结: 优质健康文件: {GREEN}{clean_count}{RESET} 个 | 建议剔除的毒瘤文件: {RED if danger_count > 0 else GREEN}{danger_count}{RESET} 个{RESET}")
    if danger_count > 0:
        print(f"{RED}💡 提示: 请立刻把标红(☢️)和标黄(⚠️)的文件移出训练集，它们就是卡死 5.5 梯度的罪魁祸首！{RESET}")
    print("-" * 85 + "\n")


if __name__ == "__main__":
    # 填入你当前的真实实验数据集路径
    DATASET_PATH = "/home/jue/文档/dataset/temp-dataset/"
    run_scan(DATASET_PATH)