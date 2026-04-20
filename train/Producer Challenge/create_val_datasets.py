import os
import random
import shutil
import re

def split_by_percentage(test_ratio, pattern=r'(.+?)_\d+\.'):
    """
    按数据集分组百分比抽样
    :param test_ratio: 每个数据集的测试集百分比
    :param pattern: 文件名匹配正则
    """
    base_dir = os.getcwd()
    img_src = os.path.join(base_dir, 'images', 'Train')
    label_src = os.path.join(base_dir, 'labels', 'Train')
    
    # 创建测试目录
    img_dst = os.path.join(base_dir, 'images', 'val')
    label_dst = os.path.join(base_dir, 'labels', 'val')
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(label_dst, exist_ok=True)

    # 数据集分组
    dataset_dict = {}
    for img_file in os.listdir(img_src):
        match = re.match(pattern, img_file)
        if not match:
            print(f"跳过无法识别的文件: {img_file}")
            continue
        
        dataset = match.group(1)
        if dataset not in dataset_dict:
            dataset_dict[dataset] = []
        dataset_dict[dataset].append(img_file)

    total_test = 0
    for dataset, files in dataset_dict.items():
        # 计算应抽取数量（至少1个）
        total = len(files)
        sample_num = max(1, round(total * test_ratio / 100))

        # 实际抽样数量处理
        actual_sample = min(sample_num, total)
        if actual_sample == 0:
            print(f"数据集 {dataset} 跳过（计算样本数为0）")
            continue

        # 排序后均匀间隔抽样，避免集中抽取
        sorted_files = sorted(files)
        interval = total / actual_sample
        selected = [sorted_files[int(interval * i + interval / 2)] for i in range(actual_sample)]

        # 打乱顺序，避免按文件名顺序处理
        random.shuffle(selected)
        
        # 移动文件
        moved_imgs = 0
        moved_labels = 0
        for img_file in selected:
            # 处理图片
            src_img = os.path.join(img_src, img_file)
            dst_img = os.path.join(img_dst, img_file)
            if os.path.exists(src_img) and not os.path.exists(dst_img):
                shutil.move(src_img, dst_img)
                moved_imgs += 1

            # 处理标签
            label_file = f"{os.path.splitext(img_file)[0]}.txt"
            src_label = os.path.join(label_src, label_file)
            dst_label = os.path.join(label_dst, label_file)
            if os.path.exists(src_label) and not os.path.exists(dst_label):
                shutil.move(src_label, dst_label)
                moved_labels += 1

        total_test += moved_imgs
        print(f"数据集 [{dataset}] 原样本 {total} → 测试集 {moved_imgs} 张图片, {moved_labels} 个标签 ({100*moved_imgs/total:.1f}%)")

    print(f"\n总测试样本：{total_test}")

if __name__ == '__main__':
    # 交互输入
    while True:
        try:
            ratio = float(input("请输入每个数据集的测试集百分比(0.1-99.9): "))
            if 0.1 <= ratio <= 99.9:
                break
            print("请输入0.1到99.9之间的数值")
        except ValueError:
            print("输入无效，请重新输入")
    
    # 高级模式设置
    custom_pattern = input("可选：输入自定义文件名模式正则(直接回车使用默认)：")
    
    split_by_percentage(
        test_ratio=ratio,
        pattern=custom_pattern.strip() if custom_pattern else r'(.+?)_\d+\.'
    )
