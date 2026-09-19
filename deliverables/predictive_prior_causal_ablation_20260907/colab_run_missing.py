from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
root=Path('/content/PredictivePriorCausalAblation')
for seed in (314,2718):
    output=Path(f'/content/predictive_prior_causal_seed_{seed}')
    archive=output.with_suffix('.zip')
    if archive.is_file():
        print(json.dumps({'seed':seed,'status':'already_complete'}),flush=True); continue
    cmd=[sys.executable,'-m','implementations.lite_globe.run_predictive_prior_causal_ablation','--seed',str(seed),'--teacher-dir',str(root/'ResearchAIWorkspace/artifacts/lite_globe/phase7/checkpoints'),'--legacy-dir',str(root/'ResearchAIWorkspace/artifacts/lite_globe/phase11/checkpoints'),'--output-dir',str(output),'--device','cuda','--train-episodes-per-scenario-rollout','100','--evaluation-episodes','200','--epochs','120','--batch-size','512','--learning-rate','0.005','--warmup','50','--latency-repeats','2000','--zip-results']
    p=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
    (Path(f'/content/predictive_prior_causal_seed_{seed}.log')).write_text(p.stdout+'\n--- STDERR ---\n'+p.stderr)
    if p.returncode: raise RuntimeError(f'seed {seed} failed')
    print(json.dumps({'seed':seed,'status':'complete','zip':str(archive)}),flush=True)
print(json.dumps({'complete':True}),flush=True)
