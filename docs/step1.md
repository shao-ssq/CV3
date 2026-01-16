### 权重文件对比

| 项目 | 权重文件 | 内容 | vllm 是否加载 |
|------|---------|------|--------------|
| CosyVoice2 | llm.pt | Qwen2 + speech_embedding + llm_decoder | ✅ 全部加载 |
| CosyVoice3 | model.safetensors | 只有 Qwen2 | ✅ 加载 |
| CosyVoice3 | llm.pt | speech_embedding + llm_decoder | ❌ **未加载** |

### vllm 权重加载流程

```python
# vllm_use_cosyvoice2_model.py 的 load_weights 函数
def load_weights(self, weights):
    weights = self.convert_weights(weights)  # 转换权重名称
    loader = AutoWeightsLoader(self)
    loader.load_weights(weights)  # 只从 safetensors 加载
```

**问题**: `AutoWeightsLoader` 只加载 `model.safetensors`，不会加载 `llm.pt`！

### 结果

- `speech_embedding`: 随机初始化 (未加载训练权重)
- `llm_decoder`: 随机初始化 (未加载训练权重)
- 模型无法正确处理 speech tokens → 生成垃圾输出

---

## 七、修复方案: 合并权重文件

将 `llm.pt` 中的关键权重合并到 `model.safetensors`:

```bash
python3 << 'EOF'
import torch
from safetensors.torch import load_file, save_file

model_dir = '/root/PycharmProjects/CV3/pretrained_models/Fun-CosyVoice3-0.5B'

# 加载现有权重
sf_state = load_file(f'{model_dir}/model.safetensors')
llm_state = torch.load(f'{model_dir}/llm.pt', map_location='cpu', weights_only=True)

# 合并 speech_embedding 和 llm_decoder
for key in ['speech_embedding.weight', 'llm_decoder.weight']:
    if key in llm_state:
        sf_state[key] = llm_state[key]
        print(f'Added: {key} -> {llm_state[key].shape}')

# 保存合并后的权重
save_file(sf_state, f'{model_dir}/model.safetensors')
print('Done!')
EOF
```

