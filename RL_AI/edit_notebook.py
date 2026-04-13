import json
from pathlib import Path

filepath = r'c:\code\capstone-temp\RL_AI\start.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

new_cell = {
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
       'import subprocess\n',
       'import sys\n',
       'from pathlib import Path\n',
       '\n',
       'print("=== 밸런싱 풀-크로스 검증 실험 (800판 자동 로깅) 시작 ===")\n',
       '\n',
       '# DLPC 환경과 로컬 환경 호환을 위한 경로 설정\n',
       'home = Path.home()\n',
       'rl_ai_root = home / "RL_AI"\n',
       'if not rl_ai_root.exists():\n',
       '    rl_ai_root = Path.cwd() # 로컬 실행 시 현재 폴더 사용\n',
       '\n',
       'runner_path = rl_ai_root / "match_runner.py"\n',
       'model_path = rl_ai_root / "my_best_model.pt"\n',
       '\n',
       '# --comprehensive 플래그를 사용하여 8가지 조합 x 100판 = 800판을 모두 섭렵합니다.\n',
       '# P1, P2 역할과 덱은 스크립트가 알아서 골고루 자동 스왑합니다!\n',
       'cmd = [\n',
       '    sys.executable, str(runner_path),\n',
       '    "--p1_model", str(model_path),\n',
       '    "--comprehensive",\n',
       '    "--matches", "100"\n',
       ']\n',
       '\n',
       '# DLPC의 기본 루트 경로(home)를 기준으로 실행 (로컬이면 로컬 cwd)\n',
       'subprocess.run(cmd, cwd=str(home if (home/"RL_AI").exists() else Path.cwd()))\n',
       '\n',
       'print("=== 밸런싱 실험 완벽하게 종료! ===")\n',
       'print("생성된 로그를 변환하려면 터미널에서 balance_logger.py를 실행해 주세요.")\n'
   ]
}

notebook['cells'].append(new_cell)

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print('Cell appended successfully.')
