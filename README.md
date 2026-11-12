# 手写数字识别 · 神经网络可视化 / Handwritten Digit Neural Network Visualizer

[中文](#中文) | [English](#english)

---

## 中文

用鼠标（或手指）写一个 0–9 的数字，看一个小型神经网络如何一层一层地“亮起来”，从像素一路算到答案。

**使用方法：** 下载 `index.html`，用浏览器直接打开即可。不用联网，不用安装任何东西，所有计算都在浏览器本地完成。

### 功能
- 手写画板，松手后自动识别
- 逐层动画：输入层 → 隐藏层 1 → 隐藏层 2 → 输出层，每一步都有文字讲解
- 连线颜色表示贡献方向：橙色推高、蓝色压低，线越粗贡献越大
- 点击任意神经元，查看它的权重模板（它在“找”什么样的笔画）
- 动画速度可调（慢放 / 正常 / 快速）
- 中英文界面一键切换

### 模型
- 全连接网络：784 → 64 (ReLU) → 32 (ReLU) → 10 (softmax)，共 52,544 个权重
- 训练数据：程序生成的手写笔画 + 字体渲染 + 部分真实手写样本，并做了旋转、倾斜、弹性形变等数据增强
- 在留出的真实手写测试集上准确率约 98%
- 权重量化为 int8 后直接嵌入网页（约 100 KB）

### 自己重新训练
需要 Python 3、numpy、scipy、Pillow、scikit-learn：
```bash
cd training
python gen.py 60000   # 生成合成数据
python mix.py         # 混入真实手写样本
python train.py       # 用 numpy 训练（几分钟）
python build.py       # 把权重写进 ../index.html
```

---

## English

Draw a digit 0–9 with your mouse (or finger) and watch a small neural network light up layer by layer, from raw pixels to the final answer.

**How to use:** download `index.html` and open it in any browser. No internet, no install; everything runs locally.

### Features
- Drawing canvas with automatic recognition when you release
- Layer-by-layer animation (input → hidden 1 → hidden 2 → output) with an explanation at each step
- Connection colors show direction: orange pushes up, blue pushes down; thicker means a bigger contribution
- Click any neuron to see its weight template (the stroke pattern it looks for)
- Adjustable animation speed (slow / normal / fast)
- One-click Chinese / English switch

### Model
- Fully connected network: 784 → 64 (ReLU) → 32 (ReLU) → 10 (softmax), 52,544 weights
- Training data: procedurally generated strokes, font renderings and some real handwriting, augmented with rotation, shear and elastic distortion
- About 98% accuracy on held-out real handwriting
- Weights are quantized to int8 and embedded in the page (~100 KB)

### Retrain it yourself
Requires Python 3, numpy, scipy, Pillow, scikit-learn:
```bash
cd training
python gen.py 60000   # generate synthetic data
python mix.py         # mix in real handwriting samples
python train.py       # train with plain numpy (a few minutes)
python build.py       # write the weights into ../index.html
```

---

Made by INLEI
