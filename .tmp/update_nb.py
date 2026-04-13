import json
import pathlib

file_path = pathlib.Path(r'c:\code\capstone-temp\RL_AI\start.ipynb')
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Append code to the last cell
append_code = [
    '\n# 모델 저장 코드 추가\n',
    'agent.save(rl_ai_root / \"my_best_model.pt\")\n',
    'print(f\"\\n[Model Saved] => {rl_ai_root / \'my_best_model.pt\'}\")\n'
]
nb['cells'][-1]['source'].extend(append_code)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
