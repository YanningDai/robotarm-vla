from pathlib import Path
import yaml
from datetime import datetime

def extract_params_from_vla_load_path(vla_load_path):
    """从 vla_load_path 中提取 rollback_alpha, trust_region_delta, seed"""
    # 路径格式: ../SimplerEnv/outputs/ppo,truly,rollback_alpha=1.0,trust_region_delta=0.05,seed=0/steps_0239
    # 需要提取: ppo,truly,rollback_alpha=1.0,trust_region_delta=0.05,seed=0
    
    path_parts = vla_load_path.rstrip("/").split("/")
    if len(path_parts) < 2:
        return "", None
    
    params_str = path_parts[-2]  # 倒数第二个就是参数部分
    parts = params_str.split(",")
    
    params = {}
    seed = None
    
    for part in parts:
        if part.startswith("rollback_alpha="):
            params["rollback_alpha"] = part.split("=")[1]
        elif part.startswith("trust_region_delta="):
            params["trust_region_delta"] = part.split("=")[1]
        elif part.startswith("seed="):
            seed = part.split("=")[1]
    
    param_str = ",".join([f"{k}={v}" for k, v in sorted(params.items())])
    return param_str if param_str else "", seed

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
        params, seed = extract_params_from_vla_load_path(vla_load_path)
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

    # 允许空文件；解析失败时当作空 dict
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

            for params, seeds_data in split_data.items():
                # 允许 params 为 ""（例如路径没解析出参数）
                all_params.add(params)
                data.setdefault(params, {})
                data[params].setdefault(task_name, {})

                # --- 这里的策略是：只用 seed "0"；如果想换成“若无0则取任意一个seed”或“对所有seed求平均”，看注释 ---
                if isinstance(seeds_data, dict) and "0" in seeds_data and isinstance(seeds_data["0"], dict):
                    data[params][task_name] = seeds_data["0"]
                else:
                    # 1) 若无 seed "0"，保持留空（默认策略）
                    pass

                    # 2) 【可选策略A】若无 seed "0"，取任意一个 seed：
                    # if isinstance(seeds_data, dict) and seeds_data:
                    #     first_seed_key = sorted(seeds_data.keys(), key=lambda x: (str(x))) [0]
                    #     if isinstance(seeds_data[first_seed_key], dict):
                    #         data[params][task_name] = seeds_data[first_seed_key]

                    # 3) 【可选策略B】对所有 seed 的某 metric（比如 success）求平均：
                    # if isinstance(seeds_data, dict) and seeds_data:
                    #     keys = [k for k in seeds_data.keys() if isinstance(seeds_data[k], dict)]
                    #     if keys:
                    #         # 以 success 为例
                    #         vals = [seeds_data[k].get("success") for k in keys if isinstance(seeds_data[k].get("success"), (int, float))]
                    #         if vals:
                    #             data[params][task_name] = dict(success=sum(vals)/len(vals))

    all_params = sorted(list(all_params))
    all_tasks = sorted(list(all_tasks))

    # 计算“每个参数的平均 success”（跨 task 平均）——若无数据则留空字符串
    param_avg_success = {}
    for param in all_params:
        success_values = []
        for task in all_tasks:
            metrics = data.get(param, {}).get(task, {})
            val = metrics.get("success")
            if isinstance(val, (int, float)):
                success_values.append(val)
        param_avg_success[param] = (sum(success_values) / len(success_values)) if success_values else ""

    metrics = ["consecutive_grasp", "is_src_obj_grasped", "success"]

    # 如果 task 为空，我们也仍然写 CSV，只包含 "params" 和 "avg_success_rate" 表头
    base_dir = stats_file.parent
    for metric in metrics:
        csv_path = base_dir / f"statistics_{metric}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            if all_tasks:
                header = ["params"] + all_tasks + ["avg_success_rate"]
            else:
                header = ["params", "avg_success_rate"]  # 没有任务列时的最小表头
            f.write(",".join(header) + "\n")

            # 如果没有任何 params，也不写数据行；只保留表头即可
            for param in all_params:
                # params 里的逗号 -> 分号，免得和 CSV 分隔符冲突
                param_display = (param or "").replace(",", ";")
                row = [param_display]

                # 逐 task 填值；无值写空字符串
                for task in all_tasks:
                    metrics_dict = data.get(param, {}).get(task, {})
                    val = metrics_dict.get(metric, "")
                    row.append(str(val) if isinstance(val, (int, float)) else ("" if val is None else str(val)))

                # 追加平均 success（不依赖当前 metric；根据上面 param_avg_success）
                avg_success = param_avg_success.get(param, "")
                row.append(str(avg_success) if isinstance(avg_success, (int, float)) else "")

                f.write(",".join(row) + "\n")

        print(f"Generated {csv_path} ")
        
if __name__ == "__main__":
    stats_file = main()  
    generate_statistics_tables(stats_file)