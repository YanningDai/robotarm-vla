import json
from collections import defaultdict
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from scipy.special import lambertw
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, BatchFeature
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class OpenVLAPolicy:
    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        self.tpdv = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.bfloat16)
        self.tpdv_vn = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.float32)
        self.action_scale = 1.0

        # openvla: register
        self.image_processor = PrismaticImageProcessor.from_pretrained(self.args.vla_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.vla_path, trust_remote_code=True, padding_side="left")
        self.processor = PrismaticProcessor.from_pretrained(
            self.args.vla_path,
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            trust_remote_code=True
        )
        # self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )

        # openvla: lora
        if not self.args.vla_load_path:
            lora_config = LoraConfig(
                r=self.args.vla_lora_rank,
                lora_alpha=min(self.args.vla_lora_rank, 16),
                lora_dropout=0.0,
                target_modules=[
                    "proj", "qkv", "fc1", "fc2",  # vision
                    "q", "kv", "fc3",  # project
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",  # llm
                ],
                init_lora_weights="gaussian"
            )
            self.vla = get_peft_model(self.vla, lora_config)
        else:
            self.vla = PeftModel.from_pretrained(self.vla, self.args.vla_load_path, is_trainable=True)
            print(f"VLA load: {self.args.vla_load_path}")

            if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
                path = Path(self.args.vla_load_path) / "dataset_statistics.json"
                ds = json.load(open(path, "r"))
                self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        # set value head trainable
        for name, param in self.vla.named_parameters():
            if "value_head" in name:
                param.requires_grad = True

        self.vla.print_trainable_parameters()

        # openvla: optimizer
        self.params_vh = None
        self.params_vla = None
        self.vh_optimizer = None
        self.vla_optimizer = None
        self._setup_optimizer()

        if self.args.vla_load_path:
            training_state_path = Path(self.args.vla_load_path) / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path, map_location=self.tpdv["device"])

                if "vh" in training_state:
                    self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
                else:
                    print("Warning: value_head state not found in training_state")

                self._setup_optimizer()
                self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
                self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

                print(f"Optimizer load: {self.args.vla_load_path}")
            else:
                print(f"Warning: training_state not found in {training_state_path}")

    def _setup_optimizer(self):
        self.params_vh = [p for n, p in self.vla.named_parameters() if "value_head" in n and p.requires_grad]
        self.params_vla = [p for n, p in self.vla.named_parameters() if "value_head" not in n and p.requires_grad]
        betas = (self.args.vla_optim_beta1, self.args.vla_optim_beta2)
        self.vh_optimizer = AdamW(self.params_vh, lr=self.args.vla_vhlr, betas=betas)
        self.vla_optimizer = AdamW(self.params_vla, lr=self.args.vla_lr, betas=betas)

    def _preprocess_obs(self, x: dict, action: torch.Tensor = None) -> BatchFeature:
        images = x["image"]
        task_description = x["task_description"]

        assert isinstance(images, torch.Tensor)
        assert len(images.shape) == 4
        assert images.shape[3] == 3
        assert images.dtype == torch.uint8

        assert isinstance(task_description, list)
        assert isinstance(task_description[0], str)
        assert images.shape[0] == len(task_description)

        images = images.permute(0, 3, 1, 2)  # [B, C, H, W]
        images = images.to(**self.tpdv)

        # prompt
        if action is None:
            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: "
                           for t in task_description]
        else:
            assert isinstance(action, torch.Tensor)
            # action = action.cpu().numpy() # [B, dim]
            action_str = self.tokenizer.batch_decode(action)

            task_prompt = [f"In: What action should the robot take to {t.lower()}?\nOut: {a}</s>"
                           for t, a in zip(task_description, action_str)]

        inputs = self.processor(task_prompt, images, padding=True)
        inputs = inputs.to(**self.tpdv)

        if action is not None:
            inputs["labels"] = inputs["input_ids"].clone()

        return inputs

    def get_action(self, x: dict, deterministic) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        do_sample = (temperature != 0.0)
        features = self._preprocess_obs(x)

        values, action, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
        )

        assert len(values.shape) == 2 and values.shape[1] == 1
        assert len(action.shape) == 2 and action.shape[0] == values.shape[0]
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return values, action, logprobs

    def get_action_temp(self, x: dict, do_sample, temperature, num_beams) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x)

        _, action, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.args.vla_unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams,
        )

        assert len(action.shape) == 2
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return action, logprobs

    def get_value(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)

        value = self.vla.get_value(**features)

        assert len(value.shape) == 2 and value.shape[1] == 1

        return value

    def get_hidden(self, x: dict) -> torch.Tensor:
        features = self._preprocess_obs(x)

        hs = self.vla.get_hidden(**features)

        return hs

    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._preprocess_obs(x, action)

        logprobs, entropy, values = self.vla.evaluate_action(
            **features,
            unnorm_key=self.args.vla_unnorm_key
        )

        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1
        assert len(entropy.shape) == 2 and entropy.shape[1] == 1
        assert len(values.shape) == 2 and values.shape[1] == 1

        return logprobs, entropy, values

    def prep_rollout(self):
        self.vla.eval()

    def prep_training(self):
        self.vla.train()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

        self.vla.save_pretrained(str(path))
        training_state = {
            "vh": self.vla.value_head.state_dict(),
            "vh_optimizer": self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
        }
        torch.save(training_state, path / "training_state.pt")

        json.dump(self.vla.base_model.norm_stats, open(path / "dataset_statistics.json", "w"))

    def load(self, path: Path):
        del self.vla
        torch.cuda.empty_cache()

        self.vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )
        self.vla = PeftModel.from_pretrained(self.vla, path, is_trainable=True)
        self.vla.print_trainable_parameters()

        if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
            ds = json.load(open(path / "dataset_statistics.json", "r"))
            self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        training_state_path = path / "training_state.pt"
        training_state = torch.load(training_state_path, map_location=self.tpdv["device"])

        if "vh" in training_state:
            self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
        else:
            print("Warning: value_head state not found in training_state")

        self._setup_optimizer()
        self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
        self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

class OpenVLAPPO:
    def __init__(self, all_args, policy: OpenVLAPolicy):
        self.args = all_args
        self.policy = policy
        self.ppo_clip = 0.2
        self.ppo_grad_norm = 10.0
        self.ppo_entropy_coef = self.args.alg_entropy_coef
        self.ppo_huber_delta = 10.0
        self.tpdv = self.policy.tpdv
        self.tpdv_vn = self.policy.tpdv_vn
        self.trust_type = self.args.trust_type

    def train_ppo_step(self, idx, total, batch):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        # --------------------- preprocess ---------------------
        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)  # uint8
        actions = torch.tensor(actions).to(self.tpdv["device"])  # int32
        value_preds = torch.tensor(value_preds).to(**self.tpdv)
        returns = torch.tensor(returns).to(**self.tpdv_vn)  # float32
        # masks = torch.tensor(masks).to(**self.tpdv)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)
        returns_norm = returns.to(**self.tpdv)
        mb_advantage = advantages
        mb_logps = old_logprob
        # ---------------------- Policy loss ----------------------
        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)
        new_logps = logprob
        logprobs_diff = new_logps - mb_logps
        ratio = torch.exp(logprobs_diff)
        surr1 = ratio * mb_advantage
        approx_kl = None
        
        # =========================================================
        # Policy loss (trust region variants)
        # =========================================================

        pg_losses = -mb_advantage * ratio

        if self.args.trust_type == "clip":
            pg_losses2 = -mb_advantage * torch.clamp(
                ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange
            )
            pg_loss = torch.max(pg_losses, pg_losses2)

        elif self.args.trust_type == "dual_clip":
            pg_losses2 = -mb_advantage * torch.clamp(
                ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange
            )
            pg_losses_dual = -mb_advantage * self.args.cliprange_dual
            pg_loss = torch.min(torch.max(pg_losses, pg_losses2), pg_losses_dual)

        elif self.args.trust_type == "clip_high":
            pg_losses2 = -mb_advantage * torch.clamp(
                ratio,
                1.0 - self.args.cliprange_low,
                1.0 + self.args.cliprange_high,
            )
            pg_loss = torch.max(pg_losses, pg_losses2)

        elif self.args.trust_type == "dynamic_clip":
            # mb_logps: old logps
            mb_ps = torch.exp(mb_logps)

            low_c_ref = torch.where(mb_ps != 0, 4 * self.args.cliprange_low / (mb_ps), 0)
            low_c_ref = 1 - low_c_ref
            low_c_ref = low_c_ref * (low_c_ref >= 0)
                                    
            high_c_ref = 4 * self.args.cliprange_high / mb_ps
            high_c_ref = torch.where(mb_ps != 0, 1 + 4 * self.args.cliprange_high / mb_ps, float("inf"))

            low = (1 + torch.sqrt(low_c_ref)) / 2
            high = (1 + torch.sqrt(high_c_ref)) / 2
            
            clip_ratio_c = max(self.args.cliprange_dual, 10)

            dynamic_cliprange_low = low
            dynamic_cliprange_high = torch.clamp(high, 0, clip_ratio_c)

            pg_losses2 = -mb_advantage * torch.clamp(
                ratio, 1.0 - dynamic_cliprange_low, 1.0 + dynamic_cliprange_high
            )
            pg_loss = torch.max(pg_losses, pg_losses2)

        elif self.args.trust_type == "clip_cov":
            clip_cov_ratio = self.args.clip_cov_ratio   # e.g. 0.002
            clip_cov_lb = self.args.clip_cov_lb         # e.g. 1.0
            clip_cov_ub = self.args.clip_cov_ub         # e.g. 5.0
            corr = torch.ones_like(mb_advantage)
            clip_by_origin = (
                (ratio < 1.0 - self.args.cliprange) |
                (ratio > 1.0 + self.args.cliprange)
            )
            cov_all = (mb_advantage - mb_advantage.mean()) * (
                logprobs_diff - logprobs_diff.mean()
            )
            cov_all[clip_by_origin] = -torch.inf
            clip_num = max(int(clip_cov_ratio * mb_advantage.numel()), 1)
            top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb)
            top_k_idx = torch.nonzero(top_k_idx).squeeze(-1)
            
            if len(top_k_idx) > 0:
                perm = torch.randperm(len(top_k_idx), device=top_k_idx.device)
                top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
            else:
                top_k_idx = torch.empty(0, device=cov_all.device, dtype=torch.long)
            
            corr[top_k_idx] = 0

            pg_losses2 = -mb_advantage * torch.clamp(
                ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange
            )
            pg_loss = torch.max(pg_losses, pg_losses2) * corr

        elif self.args.trust_type == "soft_gated":
            tau_pos = self.args.tau_pos      # e.g. 1.0
            tau_neg = self.args.tau_neg      # e.g. 1.05

            coef1 = ratio               # 对应截图里的 coef1

            gate_pos = torch.sigmoid(tau_pos * (coef1 - 1.0))
            gate_neg = torch.sigmoid(tau_neg * (coef1 - 1.0))

            soft_gate = torch.where(mb_advantage > 0, gate_pos, gate_neg)

            pg_loss = -soft_gate * mb_advantage * ratio

        elif self.args.trust_type == "trgppo":
            delta = self.args.delta
            ROOT = Path(__file__).resolve().parents[2]
            if str(ROOT) not in sys.path:
                sys.path.append(str(ROOT))
                sys.path.append(str(ROOT)+"/utils")
            try:
                from utils.KL2Clip_discrete import KL2Clip
                print("Successfully imported KL2Clip")
            except ImportError as e:
                print(f"Failed to import KL2Clip: {e}")
            kl2clip = KL2Clip()
            old_per_token_ps = torch.exp(mb_logps)
            old_per_token_ps_array = old_per_token_ps.detach().float().cpu().view(-1).numpy()
            results = kl2clip(old_per_token_ps_array, delta, decimal_places=4)
            kl2clip_low = torch.from_numpy(results.min.reshape(old_per_token_ps.shape)).to(ratio.device, ratio.dtype)
            kl2clip_high = torch.from_numpy(results.max.reshape(old_per_token_ps.shape)).to(ratio.device, ratio.dtype)
            pg_loss = -mb_advantage * torch.clamp(ratio, kl2clip_low, kl2clip_high)
            
        elif self.args.trust_type == "trgc":
            delta = self.args.delta
            
            z = -np.exp(-(delta + 1.0))
            c_low = float(-lambertw(z, k=0).real)
            c_high = float(-lambertw(z, k=-1).real)
            
            # temp
            # if abs(delta - 0.07) < 1e-3:
            #     c_low, c_high = 0.07, 1.0
            # elif abs(delta - 0.31) < 1e-3:
            #     c_low, c_high = 0.31, 0.45
            # elif abs(delta - 0.284) < 1e-3:
            #     c_low, c_high = 0.284, 0.539
            # else:
            #     raise ValueError(f"Unsupported delta: {delta}")
    
            pg_losses2 = -mb_advantage * torch.clamp(
                ratio,
                1.0 - c_low,
                1.0 + c_high,
            )
            pg_loss = torch.max(pg_losses, pg_losses2)
            
        else:
            raise ValueError(f"Unknown trust_type: {self.args.trust_type}")

        policy_loss = pg_loss.sum(dim=-1).mean()
        
        # ----------------------- Value loss -----------------------
        value_pred_clipped = value_preds + (values - value_preds).clamp(-self.ppo_clip, self.ppo_clip)
        error_clipped = returns_norm - value_pred_clipped
        error_original = returns_norm - values
        value_loss_clipped = huber_loss(error_clipped, self.ppo_huber_delta)
        value_loss_original = huber_loss(error_original, self.ppo_huber_delta)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_clip_indicator = (value_pred_clipped - value_preds).abs() > self.ppo_clip
        value_clip_ratio = value_clip_indicator.to(**self.tpdv).mean()

        value_loss = value_loss.mean()

        # ----------------------- Entropy loss -----------------------
        entropy_loss = entropy.mean()

        # ------------------------ Total loss ------------------------
        loss = policy_loss + value_loss - self.ppo_entropy_coef * entropy_loss
        loss /= self.args.alg_gradient_accum
        loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla + self.policy.params_vh, self.ppo_grad_norm)
            self.policy.vh_optimizer.step()
            self.policy.vla_optimizer.step()
            self.policy.vh_optimizer.zero_grad()
            self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
            ratio=ratio.mean().item(),
            ratio_median=ratio.median().item(),
            ratio_2=(mb_logps - old_logprob).mean().exp().item(),

            value_clip_ratio=value_clip_ratio.item(),
            value_old_mean=value_preds.mean().item(),
            values_mean=values.mean().item(),
            returns_mean=returns.mean().item(),
            returns_norm_mean=returns_norm.mean().item(),
            logprob_mean=mb_logps.mean().item(),
            logprob_old_mean=old_logprob.mean().item(),
        )
        
        if approx_kl is not None:
            info["approx_kl"] = approx_kl.mean().item()
            
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()

        return info

    def train_grpo_step(self, idx, total, batch):
        obs_image, instruct, actions, value_preds, returns, masks, old_logprob, advantages = batch

        obs = dict(image=torch.tensor(obs_image).to(self.tpdv["device"]), task_description=instruct)  # uint8
        actions = torch.tensor(actions).to(self.tpdv["device"])  # int32
        # value_preds = torch.tensor(value_preds).to(**self.tpdv)
        # returns = torch.tensor(returns).to(**self.tpdv_vn) # float32
        # masks = torch.tensor(masks).to(**self.tpdv)
        old_logprob = torch.tensor(old_logprob).to(**self.tpdv)
        advantages = torch.tensor(advantages).to(**self.tpdv)

        # Policy loss
        logprob, entropy, values = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(logprob - old_logprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).sum(dim=-1, keepdim=True).mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = policy_loss - self.ppo_entropy_coef * entropy_loss
        loss /= self.args.alg_gradient_accum
        loss.backward()

        if idx % self.args.alg_gradient_accum == (self.args.alg_gradient_accum - 1) or idx == (total - 1):
            grad_norm = nn.utils.clip_grad_norm_(self.policy.params_vla, self.ppo_grad_norm)
            self.policy.vla_optimizer.step()
            self.policy.vla_optimizer.zero_grad()
        else:
            grad_norm = None

        info = dict(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            entropy_loss=entropy_loss.item(),
            ratio=ratio.mean().item(),
            ratio_median=ratio.median().item(),
            ratio_2=(logprob - old_logprob).mean().exp().item(),

            logprob_mean=logprob.mean().item(),
            logprob_old_mean=old_logprob.mean().item(),
        )
        if grad_norm is not None:
            info["grad_norm"] = grad_norm.item()

        return info

    def train_ppo(self, buffer):
        train_info = defaultdict(lambda: [])

        # buffer
        buffer.compute_returns_ppo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()

            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_ppo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)

        return final_info

    def train_grpo(self, buffer):
        train_info = defaultdict(lambda: [])

        # buffer
        buffer.compute_returns_grpo()
        minibatch_count = buffer.get_minibatch_count()

        for _ in range(self.args.alg_ppo_epoch):
            data_generator = buffer.feed_forward_generator()

            for idx, batch in tqdm(enumerate(data_generator), total=minibatch_count, desc="train"):
                info = self.train_grpo_step(idx, minibatch_count, batch)
                for key, value in info.items():
                    train_info[key].append(value)

        final_info = {}
        for key, value in train_info.items():
            final_info[key] = np.mean(value)

        return final_info
