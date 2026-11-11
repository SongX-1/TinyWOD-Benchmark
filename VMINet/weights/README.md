# 本地预训练权重

默认 B 只需把 **B-ckpt.pth.tar** 放在本目录，即 `<项目根目录>/weights/B-ckpt.pth.tar`。保持权重文件本身即可，不必解压成项目目录。

用浏览器手动获取。地址来自作者原始 README：

| 变体 | 作者原文给出的地址 | 默认本地路径 |
|---|---|---|
| B（本次默认） | https://github.com/yws-wxs/VMINet/releases/download/4.22/B-ckpt.pth.tar | weights/B-ckpt.pth.tar |
| Ti | https://github.com/yws-wxs/VMINet/releases/download/output/VMINet2.0-Ti-checkpoint.pth.tar | weights/VMINet2.0-Ti-checkpoint.pth.tar |
| XS | https://github.com/yws-wxs/VMINet/releases/download/output/VMINet2.0-XS-checkpoint.pth.tar | weights/VMINet2.0-XS-checkpoint.pth.tar |
| S | https://github.com/yws-wxs/VMINet/releases/download/output/VMINet2.0-S-checkpoint.pth.tar | weights/VMINet2.0-S-checkpoint.pth.tar |

本轮没有获得这些官方文件，未验证链接当前可下载或实际张量兼容性。直链不可用时请到作者仓库 https://github.com/yws-wxs/VMINet 的 Releases 查找对应文件，不要把网页 HTML 改名为权重，不要把 S 改名冒充 B。

只使用信任的本地 checkpoint。原分类训练器保存含 argparse.Namespace 的文件，因此按原格式使用 torch.load(weights_only=False)，没有远程下载。

默认读取 state_dict，也支持纯参数字典。EMA 只有在 config.py 明确指定真实的 state_dict_ema 字段时使用。严格核对骨干键、形状、有限值；只忽略分类 head，空间 sum 做有记录的分辨率适配，其他差异报错。明细保存到本次 inputs/pretrained_loading.csv。

文件在其他位置只改 config.py 的 backbone_weights['b']。明确从头训练则设 pretrained=False，并记录随机初始化。完整检测器初始化使用 detector_weights，恢复训练使用 resume_from，二者优先于分类骨干权重。

训练不发起 HTTPS 请求，不自动下载，不在权重缺失后偷偷随机训练。
