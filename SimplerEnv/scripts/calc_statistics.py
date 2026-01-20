from pathlib import Path
import yaml
from datetime import datetime

# 这个代码是从wandb文件夹的offline-run里提取eval的数据
# 注意，进行之前要把不需要的wandb记录删掉

def extract_params_from_vla_load_path(vla_load_path):
    # 路径格式: ../SimplerEnv/outputs/ppo,clip_cov,para1=0.0002,para2=1.0,para3=5.0,seed=0/steps_0249
    # 需要提取: ppo,clip_cov,para1=0.0002,para2=1.0,para3=5.0,seed=0
    
    path_parts = vla_load_path.rstrip("/").split("/")
    if len(path_parts) < 2:
        return "", None
    
    params_str = path_parts[-2] # 倒数第二个就是参数部分
    parts = params_str.split(",")
    
    method_parts = parts[:2]
    params = {}
    
    for part in parts:
        if part.startswith("para1="):
            params["para1"] = part.split("=")[1]
        elif part.startswith("para2="):
            params["para2"] = part.split("=")[1]
        elif part.startswith("para3="):
            params["para3"] = part.split("=")[1]
    
    param_items = [f"{k}={v}" for k, v in sorted(params.items())]
    full_param_str = ",".join(method_parts + param_items)+'+'+ path_parts[-1]
    return full_param_str

def main():
    stats = {}
    # 记录每个 (env, split, params, seed) 最早由哪个 run 写入，用于覆盖报警
    origins = {}  # key: (env, split, params, seed) -> run_name

    wandb_path = Path(__file__).parent.parent / "wandb"
    runs = wandb_path.glob("offline-run-*")

    for run in runs:
        cfg_path = run / "glob" / "config.yaml"
        if not cfg_path.exists():
            print(f"[WARN] missing config: {cfg_path}")
            continue

        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        env_name = cfg.get("env_id", "UNKNOWN_ENV")
        vla_load_path = cfg.get("vla_load_path", "")
        params = extract_params_from_vla_load_path(vla_load_path)
        
        seed = cfg.get("seed")
        if seed is None:
            print(f"[WARN] 无法解析 seed: {vla_load_path}, run={run_name}")
            continue  # 或者设置默认值 seed = "unknown"
        run_name = run.name

        found_any = False  # 本 run 是否读到任何 stats

        # --- train ---
        train_vis = run / "glob" / "vis_0_train" / "stats.yaml"
        if train_vis.exists():
            train_stats = yaml.safe_load(train_vis.read_text()) or {}
            if "stats" in train_stats:
                found_any = True
                stats.setdefault(env_name, {}).setdefault("train", {}).setdefault(params, {})
                key = (env_name, "train", params, seed)
                if seed in stats[env_name]["train"][params]:
                    # 覆盖报警（打印上一次写入来自哪个 run）
                    prev = origins.get(key, "unknown")
                    print(f"[WARN] overwrite: env={env_name}, split=train, params='{params}', seed={seed} "
                          f"old_run='{prev}', new_run='{run_name}'")
                stats[env_name]["train"][params][seed] = train_stats["stats"]
                origins[key] = run_name

        # --- test ---
        test_vis = run / "glob" / "vis_0_test" / "stats.yaml"
        if test_vis.exists():
            test_stats = yaml.safe_load(test_vis.read_text()) or {}
            if "stats" in test_stats:
                found_any = True
                stats.setdefault(env_name, {}).setdefault("test", {}).setdefault(params, {})
                key = (env_name, "test", params, seed)
                if seed in stats[env_name]["test"][params]:
                    prev = origins.get(key, "unknown")
                    print(f"[WARN] overwrite: env={env_name}, split=test, params='{params}', seed={seed} "
                          f"old_run='{prev}', new_run='{run_name}'")
                stats[env_name]["test"][params][seed] = test_stats["stats"]
                origins[key] = run_name

        # 该 run 下确实没有任何 stats → 仅打印 EMPTY，不向 stats 里写空节点
        if not found_any:
            print(f"[EMPTY] env_id={env_name}, params='{params}', run='{run_name}' → no stats under {run/'glob'}")

    # 保存 YAML
    tt = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(__file__).parent.parent / "scripts" / "stats" / f"stats-{tt}.yaml"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False)

    return save_path

def generate_statistics_tables(stats_file: Path):
    """从 stats 文件生成统计表格。
    规则：当没有可用数据时，仍创建 CSV 并写表头，数据行留空。
    """
    if not stats_file.exists():
        print(f"[WARN] stats 文件不存在：{stats_file}")
        return

    try:
        stats_data = yaml.safe_load(stats_file.read_text()) or {}
    except Exception as e:
        print(f"[WARN] 解析 YAML 失败，当作空处理：{e}")
        stats_data = {}

    # === 汇总结构 ===
    all_params = set()
    all_tasks = set()
    data = {}   # data[params][task_name] = metrics_dict（例如 {'success': 0.8, ...}）

    # 遍历 YAML，将 train/test 的每个 env+params+seed=0 的 stats 收集
    for env_id, env_data in (stats_data or {}).items():
        if not isinstance(env_data, dict):
            continue
        for split in ["train", "test"]:
            split_data = env_data.get(split)
            if not isinstance(split_data, dict):
                continue

            task_name = f"{env_id}-{split}"
            all_tasks.add(task_name)

            ALLOWED_SEEDS = {0}
            
            for params, seeds_data in split_data.items():
                # 允许 params 为 ""（例如路径没解析出参数）
                all_params.add(params)
                data.setdefault(params, {})
                data[params].setdefault(task_name, {})

                # 对所有 seed 的 metric求平均：
                if isinstance(seeds_data, dict) and seeds_data:
                    metric_values = {}

                    for seed_key, metrics in seeds_data.items():
                        if not isinstance(metrics, dict):
                            continue
                        
                        if seed_key not in ALLOWED_SEEDS:
                            continue
    

                        for k, v in metrics.items():
                            # boolean → 0/1
                            if isinstance(v, bool):
                                metric_values.setdefault(k, []).append(1.0 if v else 0.0)
                            # numeric metric（如果以后你加了 success_rate 之类）
                            elif isinstance(v, (int, float)):
                                metric_values.setdefault(k, []).append(float(v))

                    averaged_metrics = {
                        k: (sum(vals) / len(vals)) for k, vals in metric_values.items() if vals
                    }

                    if averaged_metrics:
                        data[params][task_name] = averaged_metrics

    all_params = sorted(list(all_params))
    all_tasks = sorted(list(all_tasks))

    # 计算“每个参数的平均 success”（跨 task 平均）——若无数据则留空字符串
    param_avg_success_train = {}
    param_avg_success_test = {}
    param_avg_success_all = {}

    for param in all_params:
        train_vals = []
        test_vals = []
        all_vals = []

        for task in all_tasks:
            metrics = data.get(param, {}).get(task, {})
            val = metrics.get("success")

            if not isinstance(val, (int, float)):
                continue

            all_vals.append(val)

            if task.endswith("-train"):
                train_vals.append(val)
            elif task.endswith("-test"):
                test_vals.append(val)

        param_avg_success_train[param] = (
            sum(train_vals) / len(train_vals) if train_vals else ""
        )
        param_avg_success_test[param] = (
            sum(test_vals) / len(test_vals) if test_vals else ""
        )
        param_avg_success_all[param] = (
            sum(all_vals) / len(all_vals) if all_vals else ""
        )


    metrics = ["consecutive_grasp", "is_src_obj_grasped", "success"]

    base_dir = stats_file.parent
    for metric in metrics:
        csv_path = base_dir / f"statistics_{metric}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            if all_tasks:
                header = (
                    ["params"]
                    + all_tasks
                    + ["avg_success_train", "avg_success_test", "avg_success_all"]
                )
            else:
                header = ["params", "avg_success_rate"]  # 没有任务列时的最小表头
            f.write(",".join(header) + "\n")

            for param in all_params:
                # params 里的逗号 -> 分号，免得和 CSV 分隔符冲突
                param_display = (param or "").replace(",", ";")
                row = [param_display]

                for task in all_tasks:
                    metrics_dict = data.get(param, {}).get(task, {})
                    val = metrics_dict.get(metric, "")
                    row.append(str(val) if isinstance(val, (int, float)) else ("" if val is None else str(val)))

                # 追加平均 success（不依赖当前 metric；根据上面 param_avg_success）
                avg_train = param_avg_success_train.get(param, "")
                avg_test = param_avg_success_test.get(param, "")
                avg_all = param_avg_success_all.get(param, "")

                row.append(str(avg_train) if isinstance(avg_train, (int, float)) else "")
                row.append(str(avg_test) if isinstance(avg_test, (int, float)) else "")
                row.append(str(avg_all) if isinstance(avg_all, (int, float)) else "")

                f.write(",".join(row) + "\n")

        print(f"Generated {csv_path} ")
        
if __name__ == "__main__":
    stats_file = main()  
    generate_statistics_tables(stats_file)