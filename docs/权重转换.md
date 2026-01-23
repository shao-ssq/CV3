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

