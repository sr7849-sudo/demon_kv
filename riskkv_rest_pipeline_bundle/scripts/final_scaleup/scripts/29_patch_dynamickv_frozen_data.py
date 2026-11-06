from pathlib import Path
import shutil, datetime

p=Path("external/DynamicKV/run/longbench/pred.py")
if not p.exists():
    raise SystemExit(f"missing {p}")
s=p.read_text()

if 'parser.add_argument("--dataset", type=str, default=None' in s and "run_datasets =" in s:
    print("DynamicKV frozen-data patch already present")
    raise SystemExit(0)

backup=p.with_suffix(f".py.pre_riskkv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(p,backup)

needle='parser.add_argument("--dataset_file", type=str, default="/DynamicKV/data/LongBench")'
if needle not in s:
    raise SystemExit("Expected DynamicKV --dataset_file argument not found; inspect pred.py manually.")
s=s.replace(
    needle,
    needle + '\n    parser.add_argument("--dataset", type=str, default=None, '
             'help="Run exactly one LongBench task; RiskKV frozen-set adapter.")',
    1,
)

old = '''    for idx, dataset in enumerate(datasets):
        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")
        args.dataset = dataset
        args.data_file = f"{args.dataset_file}/{args.dataset}.jsonl"
        inference_preds(args)
'''
new = '''    run_datasets = [args.dataset] if args.dataset else datasets
    explicit_data_file = args.data_file
    for idx, dataset in enumerate(run_datasets):
        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(run_datasets)}")
        args.dataset = dataset
        if explicit_data_file:
            if len(run_datasets) != 1:
                raise ValueError("--data_file requires --dataset so one frozen file maps to one task")
            args.data_file = explicit_data_file
        else:
            args.data_file = f"{args.dataset_file}/{args.dataset}.jsonl"
        inference_preds(args)
'''
if old not in s:
    raise SystemExit("Expected bottom dataset loop not found; no patch applied.")
s=s.replace(old,new,1)
p.write_text(s)
print(f"PATCHED {p}")
print(f"BACKUP  {backup}")
