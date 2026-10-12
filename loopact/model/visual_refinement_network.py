import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from .visual_observation_encoder import VisualObservationEncoder as VisualObservationEncoder


class CrossModalFusionBlock(nn.Module):
    """
    Transformer-based feedback decoder with ResNet18 visual encoder.

    Design Philosophy:
    - model_output_action is PRIMARY signal (action features dominate)
    - Visual features provide CONTEXT/REFINEMENT (small contribution via learnable gate)
    - Transformer allows action tokens to selectively attend to relevant visual regions
    - Strong residual connection ensures action features remain dominant

    Architecture:
      1. ResNet18 encoder: obs [B,T,V,H,W,C] -> feature map [B, 512, 8, 8]
      2. Flatten to spatial tokens: [B, 64, 512]
      3. Project latent_action and obs_tokens to d_model
      4. TransformerDecoderLayer: cross-attention (action queries attend to visual keys/values)
      5. Residual: final_output = action_emb + visual_gate * (t_output - action_emb)  [95% action, 5% visual]
      6. Output head: predict velocity [B, action_horizon, action_dim]

    Why Transformer over MLP?
    - Spatial awareness: Can attend to relevant image regions per action token
    - Flexible: Different action tokens can focus on different visual areas
    - Capacity: Better at learning complex action-visual relationships
    - Already have MLP baseline, transformer adds complementary capabilities
    """

    def __init__(
        self,
        num_categories,
        input_dim,
        hidden_dim,
        output_dim,
        d_model=256,
        nhead=4,
        dim_ff=512,
        visual_gate_initial=0.30,
    ):
        """
        Args:
            num_categories: Number of embodiment categories (not used)
            input_dim: Latent action dimension (e.g., 1024)
            hidden_dim: Not used (kept for compatibility)
            output_dim: Action dimension (e.g., 7 for 7-DOF robot)
            d_model: Transformer hidden dimension (default 256)
            nhead: Number of attention heads (default 4)
            dim_ff: Feedforward dimension (default 512)
            visual_gate_initial: Initial value for visual gate (default 0.30 = 30% visual influence)
        """
        super().__init__()
        self.num_categories = num_categories
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.gate_initial = (
            visual_gate_initial  # Store initial value for reinitialization
        )

        # ResNet18 encoder: outputs [B, 512, 8, 8]
        self.obs_encoder = VisualObservationEncoder(model_name="resnet18", freeze=True)
        C_obs = 512  # ResNet18 output channels

        # Simple projection layers with Xavier initialization
        self.act_proj = nn.Linear(input_dim, d_model)
        self.obs_proj = nn.Linear(C_obs, d_model)
        self.vel_head = nn.Linear(d_model, output_dim)

        # Initialize with Xavier/Glorot uniform
        nn.init.xavier_uniform_(self.act_proj.weight)
        nn.init.zeros_(self.act_proj.bias)
        nn.init.xavier_uniform_(self.obs_proj.weight)
        nn.init.zeros_(self.obs_proj.bias)
        nn.init.xavier_uniform_(self.vel_head.weight)
        nn.init.zeros_(self.vel_head.bias)

        # Transformer decoder layer (self-attn + cross-attn + FFN)
        self.dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,  # Shapes are [B, T, D]
            dropout=0.0,
            activation="gelu",
            norm_first=True,  # Pre-norm for stability
        )

        # Learnable gate to control visual feature influence
        # This ensures model_output_action is PRIMARY, visual features provide refinement
        # Initial value is configurable via visual_gate_initial parameter
        # Lower value = more action-focused, less visual-dependent
        self.visual_gate = nn.Parameter(torch.tensor(self.gate_initial))

        # Track first forward pass to check weights after checkpoint loads
        self._first_forward_done = False

    def forward(self, model_output_action, obs, cat_ids, verbose=False):
        """
        Args:
            model_output_action: [B, action_horizon, input_dim] - latent action sequence
            obs: [B, V, T, H, W, C] - observation images (V=views, T=timesteps)
            cat_ids: [B,] - embodiment IDs (not used)
            verbose: bool - whether to print debug info (default False)
        Returns:
            output: [B, action_horizon, action_dim] - predicted velocity
        """
        # Check weights on first forward pass (checkpoint may have corrupted them)
        if not self._first_forward_done:
            self._first_forward_done = True

            # Check if weights have NaN/Inf (corrupted by checkpoint loading)
            weights_corrupted = (
                torch.isnan(self.act_proj.weight).any()
                or torch.isinf(self.act_proj.weight).any()
                or torch.isnan(self.obs_proj.weight).any()
                or torch.isinf(self.obs_proj.weight).any()
                or torch.isnan(self.vel_head.weight).any()
                or torch.isinf(self.vel_head.weight).any()
            )

            # Check visual_gate: should be ~gate_initial, if it's 0 or corrupted, reinitialize
            gate_val = self.visual_gate.item()
            gate_corrupted = (
                torch.isnan(self.visual_gate).any()
                or torch.isinf(self.visual_gate).any()
                or abs(gate_val)
                < 1e-6  # Effectively 0 (checkpoint might have set it to 0)
                # or abs(gate_val - self.gate_initial) > 0.1  # Way off from expected value
            )

            # Also check transformer decoder layer
            decoder_corrupted = False
            for param in self.dec_layer.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    decoder_corrupted = True
                    break

            if weights_corrupted or decoder_corrupted or gate_corrupted:
                print("⚠ WARNING: CrossModalFusionBlock weights corrupted by checkpoint!")
                if gate_corrupted:
                    print(
                        f"  visual_gate is corrupted/reset: {self.visual_gate.item():.6f} (expected ~{self.gate_initial})"
                    )
                print("  Reinitializing with fresh Xavier weights...")

                device = self.act_proj.weight.device
                dtype = self.act_proj.weight.dtype

                # Reinitialize with fresh tensors
                with torch.no_grad():
                    # act_proj
                    fan_in, fan_out = (
                        self.act_proj.weight.shape[1],
                        self.act_proj.weight.shape[0],
                    )
                    std = (2.0 / (fan_in + fan_out)) ** 0.5
                    self.act_proj.weight.copy_(
                        torch.randn(fan_out, fan_in, device=device, dtype=dtype) * std
                    )
                    self.act_proj.bias.zero_()

                    # obs_proj
                    fan_in, fan_out = (
                        self.obs_proj.weight.shape[1],
                        self.obs_proj.weight.shape[0],
                    )
                    std = (2.0 / (fan_in + fan_out)) ** 0.5
                    self.obs_proj.weight.copy_(
                        torch.randn(fan_out, fan_in, device=device, dtype=dtype) * std
                    )
                    self.obs_proj.bias.zero_()

                    # vel_head
                    fan_in, fan_out = (
                        self.vel_head.weight.shape[1],
                        self.vel_head.weight.shape[0],
                    )
                    std = (2.0 / (fan_in + fan_out)) ** 0.5
                    self.vel_head.weight.copy_(
                        torch.randn(fan_out, fan_in, device=device, dtype=dtype) * std
                    )
                    self.vel_head.bias.zero_()

                    # Reinitialize visual_gate to gate_initial if corrupted
                    if gate_corrupted:
                        print(
                            f"  Reinitializing visual_gate from {self.visual_gate.item():.6f} to {self.gate_initial}"
                        )
                        self.visual_gate.copy_(
                            torch.tensor(self.gate_initial, device=device, dtype=dtype)
                        )

                    # Reinitialize transformer decoder layer if corrupted
                    if decoder_corrupted:
                        print(
                            "  ⚠ Transformer decoder also corrupted - reinitializing..."
                        )
                        for name, param in self.dec_layer.named_parameters():
                            if "weight" in name and param.ndim >= 2:
                                # Xavier for weight matrices
                                if param.ndim == 2:
                                    fan_in_dec, fan_out_dec = (
                                        param.shape[1],
                                        param.shape[0],
                                    )
                                    std_dec = (2.0 / (fan_in_dec + fan_out_dec)) ** 0.5
                                    param.copy_(torch.randn_like(param) * std_dec)
                            elif "bias" in name:
                                param.zero_()

                        # Verify reinitialization worked
                        still_bad = False
                        for name, param in self.dec_layer.named_parameters():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                print(
                                    f"  ✗ WARNING: {name} still has NaN/Inf after reinit!"
                                )
                                still_bad = True

                        if still_bad:
                            print("  ⚠ Recreating transformer decoder from scratch...")
                            self.dec_layer = nn.TransformerDecoderLayer(
                                d_model=self.d_model,
                                nhead=4,
                                dim_feedforward=512,
                                batch_first=True,
                                dropout=0.0,
                                activation="gelu",
                                norm_first=True,
                            ).to(device=device, dtype=dtype)

                print(f"✓ All CrossModalFusionBlock layers reinitialized on {device}")

            # Always check and fix visual_gate if needed (even if other weights are fine)
            if gate_corrupted and not (weights_corrupted or decoder_corrupted):
                # Gate is corrupted but other weights are fine - just fix the gate
                device = self.visual_gate.device
                dtype = self.visual_gate.dtype
                with torch.no_grad():
                    print(
                        f"  Fixing visual_gate: {self.visual_gate.item():.6f} -> {self.gate_initial}"
                    )
                    self.visual_gate.copy_(
                        torch.tensor(self.gate_initial, device=device, dtype=dtype)
                    )

            # Always print visual_gate value on first forward pass for debugging
            gate_val = self.visual_gate.item()
            print(
                f"✓ visual_gate initialized: {gate_val:.6f} (visual influence: {gate_val * 100:.2f}%)"
            )

        # 1) Ensure we have a single RGB frame before permuting
        # obs can be:
        # - [B, V, T, H, W, C] (6D) -> take last view + last timestep => [B, H, W, C]
        # - [B, H, W, C] (4D) -> already a frame
        if obs.dim() == 6:
            x = obs[:, -1, -1]  # [B, H, W, C]
        elif obs.dim() == 4:
            x = obs  # [B, H, W, C]
        else:
            raise RuntimeError(
                f"Unexpected obs.dim()={obs.dim()} with shape={tuple(obs.shape)}"
            )

        # Convert to [B, 3, H, W] for ResNet18
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]

        # ResNet18 encoder -> feature map [B, 512, 8, 8]
        obs_feat = self.visual_observation_encoder(x)  # [B, 512, 8, 8]

        B, C_obs, H, W = obs_feat.shape
        # N = H * W  # Number of spatial tokens (64 for 8x8)

        # 2) Flatten to spatial tokens [B, N, C_obs]
        obs_feat = obs_feat.flatten(2).transpose(1, 2)  # [B, 64, 512]

        # 3) Project to d_model with simple Linear layers
        obs_emb = self.obs_proj(obs_feat)  # [B, 64, d_model]
        action_emb = self.act_proj(
            model_output_action
        )  # [B, T, d_model] where T is action tokens (1 or more)

        # Runtime check for NaN/extreme values (indicates bad weights)
        obs_emb_bad = (
            torch.isnan(obs_emb).any()
            or torch.isinf(obs_emb).any()
            or obs_emb.abs().max() > 1e10
        )
        action_emb_bad = (
            torch.isnan(action_emb).any() or torch.isinf(action_emb).any() or action_emb.abs().max() > 1e10
        )

        if obs_emb_bad or action_emb_bad:
            # print("\n⚠ CRITICAL: Projection outputs have extreme/NaN values!")
            # print(
            #     f"  obs_tokens range: [{obs_feat.min().item():.2f}, {obs_feat.max().item():.2f}]"
            # )
            # print(
            #     f"  model_output_action range: [{model_output_action.min().item():.2f}, {model_output_action.max().item():.2f}]"
            # )
            # print(
            #     f"  obs_emb range: [{obs_emb.min().item():.2e}, {obs_emb.max().item():.2e}]"
            # )
            # print(f"  action_emb range: [{action_emb.min().item():.2e}, {action_emb.max().item():.2e}]")
            # print(
            #     f"  obs_proj weight range: [{self.obs_proj.weight.min().item():.2e}, {self.obs_proj.weight.max().item():.2e}]"
            # )
            # print(
            #     f"  act_proj weight range: [{self.act_proj.weight.min().item():.2e}, {self.act_proj.weight.max().item():.2e}]"
            # )

            # print("\n  Attempting emergency reinitialization with fresh tensors...")
            device = self.obs_proj.weight.device
            dtype = self.obs_proj.weight.dtype

            # Create completely fresh tensors using torch.randn
            with torch.no_grad():
                # obs_proj
                fan_in, fan_out = (
                    self.obs_proj.weight.shape[1],
                    self.obs_proj.weight.shape[0],
                )
                std = (2.0 / (fan_in + fan_out)) ** 0.5
                self.obs_proj.weight.copy_(
                    torch.randn(fan_out, fan_in, device=device, dtype=dtype) * std
                )
                self.obs_proj.bias.zero_()

                # act_proj
                fan_in, fan_out = (
                    self.act_proj.weight.shape[1],
                    self.act_proj.weight.shape[0],
                )
                std = (2.0 / (fan_in + fan_out)) ** 0.5
                self.act_proj.weight.copy_(
                    torch.randn(fan_out, fan_in, device=device, dtype=dtype) * std
                )
                self.act_proj.bias.zero_()

            print(f"  Weights reinitialized on {device}, dtype={dtype}")
            print(
                f"  New obs_proj weight range: [{self.obs_proj.weight.min().item():.3f}, {self.obs_proj.weight.max().item():.3f}]"
            )
            print(
                f"  New act_proj weight range: [{self.act_proj.weight.min().item():.3f}, {self.act_proj.weight.max().item():.3f}]"
            )

            # Retry
            obs_emb = self.obs_proj(obs_feat)
            action_emb = self.act_proj(model_output_action)
            print(
                f"  After reinit - obs_emb range: [{obs_emb.min().item():.2e}, {obs_emb.max().item():.2e}]"
            )
            print(
                f"  After reinit - action_emb range: [{action_emb.min().item():.2e}, {action_emb.max().item():.2e}]"
            )

            # Final check
            obs_emb_bad = (
                torch.isnan(obs_emb).any()
                or torch.isinf(obs_emb).any()
                or obs_emb.abs().max() > 1e10
            )
            action_emb_bad = (
                torch.isnan(action_emb).any()
                or torch.isinf(action_emb).any()
                or action_emb.abs().max() > 1e10
            )
            if obs_emb_bad or action_emb_bad:
                raise RuntimeError(
                    "Projection layers still bad after reinitialization!\n"
                    "The checkpoint is incompatible with this architecture."
                )

        # 4) Transformer decoder: (self-attn on action_emb) + (cross-attn action_emb<-obs_emb) + FFN
        # This allows action tokens to attend to visual tokens, but we'll heavily weight
        # the original action features to ensure they remain primary
        t_output = self.dec_layer(action_emb, obs_emb)  # [B, 16, d_model]

        # 4.25) Stabilize transformer output to prevent gradient explosion
        # Clip extreme values and scale to match input magnitude
        t_output_max = t_output.abs().max().item()
        if t_output_max > 5.0:
            # Scale down if too large (prevents gradient explosion)
            scale_factor = 5.0 / t_output_max
            t_output = t_output * scale_factor
            if not hasattr(self, "_warned_large_output"):
                print(
                    f"⚠ WARNING: Large transformer output (max={t_output_max:.2f}), scaling by {scale_factor:.3f}"
                )
                self._warned_large_output = True

        # 4.5) Strong residual connection: action features are PRIMARY, visual features are refinement
        # Formula: final_output = action_emb + visual_gate * (t_output - action_emb)
        # visual_gate controls the balance: (1-gate)% action features, gate% visual refinement
        # This ensures model_output_action dominates, visual features provide refinement
        # The transformer learns HOW to refine, but action features are the base
        final_output = action_emb + self.visual_gate * (t_output - action_emb)  # [B, 16, d_model]

        # Check if transformer decoder produced NaN
        if torch.isnan(t_output).any() or torch.isinf(t_output).any():
            print("\n⚠ CRITICAL: Transformer decoder produced NaN!")
            print(
                f"  Input action_emb range: [{action_emb.min().item():.2f}, {action_emb.max().item():.2f}]"
            )
            print(
                f"  Input obs_emb range: [{obs_emb.min().item():.2f}, {obs_emb.max().item():.2f}]"
            )
            print(f"  Output t_output has NaN: {torch.isnan(t_output).any()}")
            print(f"  Output t_output has Inf: {torch.isinf(t_output).any()}")
            print(f"  Visual gate value: {self.visual_gate.item():.3f}")

            # Check decoder layer weights in detail
            print("\n  Checking decoder layer weights:")
            weights_bad = False
            for name, param in self.dec_layer.named_parameters():
                has_nan = torch.isnan(param).any().item()
                has_inf = torch.isinf(param).any().item()
                if has_nan or has_inf:
                    print(f"    ✗ {name}: NaN={has_nan}, Inf={has_inf}")
                    weights_bad = True
                else:
                    pmin, pmax = param.min().item(), param.max().item()
                    print(f"    ✓ {name}: range=[{pmin:.3f}, {pmax:.3f}]")

            if not weights_bad:
                print(
                    "  ⚠ Weights look OK but output is NaN - possible numerical instability!"
                )
                print(
                    "  Attempting to recreate transformer decoder with better stability..."
                )

                device = action_emb.device
                dtype = action_emb.dtype

                # Recreate with layer norm epsilon for stability
                self.dec_layer = nn.TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=4,
                    dim_feedforward=512,
                    batch_first=True,
                    dropout=0.0,
                    activation="gelu",
                    norm_first=True,
                    layer_norm_eps=1e-5,  # Default, but explicit
                ).to(device=device, dtype=dtype)

                print("  Retrying forward pass with fresh decoder...")
                t_output = self.dec_layer(action_emb, obs_emb)
                final_output = action_emb + self.visual_gate * (t_output - action_emb)

                if torch.isnan(t_output).any() or torch.isinf(t_output).any():
                    print("  ✗ Still NaN after recreation!")
                    raise RuntimeError(
                        "TransformerDecoderLayer producing NaN even after recreation!\n"
                        "This suggests a fundamental incompatibility.\n"
                        "Try training from scratch WITHOUT loading any checkpoint."
                    )
                else:
                    print(
                        f"  ✓ SUCCESS! New t_output range: [{t_output.min().item():.2f}, {t_output.max().item():.2f}]"
                    )
                    print(
                        f"  final_output range: [{final_output.min().item():.2f}, {final_output.max().item():.2f}]"
                    )
            else:
                raise RuntimeError(
                    "TransformerDecoderLayer weights still corrupted!\n"
                    "The checkpoint loading is overriding our reinitialization.\n"
                    "Solution: Train from scratch WITHOUT loading checkpoint."
                )

        # 5) Output velocity per action token
        pred_vel = self.vel_head(final_output)  # [B, 16, action_dim]

        # Debug prints (only when verbose=True)
        if verbose:
            print(f"x (prepared): {x.shape}")
            print(
                f"obs_feat (ResNet18): {obs_feat.shape}, range [{obs_feat.min().item():.2f}, {obs_feat.max().item():.2f}]"
            )
            print(
                f"model_output_action: {model_output_action.shape}, range [{model_output_action.min().item():.2f}, {model_output_action.max().item():.2f}]"
            )
            print(
                f"obs_emb: {obs_emb.shape}, range [{obs_emb.min().item():.2f}, {obs_emb.max().item():.2f}]"
            )
            print(
                f"action_emb (action features): {action_emb.shape}, range [{action_emb.min().item():.2f}, {action_emb.max().item():.2f}]"
            )
            print(
                f"t_output (transformer output): {t_output.shape}, range [{t_output.min().item():.2f}, {t_output.max().item():.2f}]"
            )
            print(
                f"visual_gate: {self.visual_gate.item():.3f} (visual influence: {self.visual_gate.item() * 100:.1f}%)"
            )
            print(
                f"final_output (action_emb + gate*(t_output-action_emb)): {final_output.shape}, range [{final_output.min().item():.2f}, {final_output.max().item():.2f}]"
            )
            print(
                f"pred_vel: {pred_vel.shape}, range [{pred_vel.min().item():.2f}, {pred_vel.max().item():.2f}]"
            )

        return pred_vel

    @property
    def visual_observation_encoder(self):
        """LoopAct API; reuse the existing checkpoint storage key."""
        return self.obs_encoder


class VisualRefinementNetwork(nn.Module):
    def __init__(self, config):
        """
        Args:
            config: Model config with action_dim, action_horizon, hidden_size, max_num_embodiments
        """
        super().__init__()
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.hidden_size = config.hidden_size

        self.action_decoder_observe = CrossModalFusionBlock(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # Track whether feedback_action is trainable
        self.tune_feedback = True

    def train(self, mode=True):
        """
        Override train() to ensure obs_encoder (ResNet18) stays in eval mode.
        ResNet18 should always be frozen, even when feedback_action is trainable.
        """
        super().train(mode)
        if mode:
            # Keep obs_encoder in eval mode even when training
            if hasattr(self.cross_modal_fusion_block, "visual_observation_encoder"):
                self.cross_modal_fusion_block.visual_observation_encoder.eval()
        return self

    def set_trainable(self, trainable: bool):
        """Set whether this module is trainable."""
        self.tune_feedback = trainable
        # Freeze/unfreeze all parameters
        self.requires_grad_(trainable)
        # Keep the frozen ResNet18 encoder frozen even when feedback_action is trainable.
        if hasattr(self.cross_modal_fusion_block, "visual_observation_encoder"):
            self.cross_modal_fusion_block.visual_observation_encoder.requires_grad_(False)
            self.cross_modal_fusion_block.visual_observation_encoder.eval()
        print(f"VisualRefinementNetwork trainable: {trainable}")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_feedback:
                # Entire feedback_action is frozen
                self.cross_modal_fusion_block.eval()
            else:
                # Only obs_encoder (ResNet18) should be frozen
                if hasattr(self.cross_modal_fusion_block, "visual_observation_encoder"):
                    self.cross_modal_fusion_block.visual_observation_encoder.eval()

    def forward(
        self,
        action_head_output: BatchFeature,
        window_idx: int,
        action_input: BatchFeature,
    ) -> BatchFeature:
        """
        Training: Run feedback decoder 16 times with 16 observations to predict 16 actions.

        At training, simple_img has 16 observations (T=16), one per action.
        Each observation predicts 1 action step.

        Args:
            action_head_output: Contains final_model [B,16,hidden], final_raw_action [B,16,action_dim], dt
            window_idx: Not used in training (processes all 16 timesteps)
            action_input: Contains simple_img [B, V=1, T=16, H, W, C], action [B,16,action_dim], action_mask

        Returns:
            BatchFeature with loss

        Each observation predicts 1 action:
        - obs[i] (timestep N+i) -> action[i]
        """
        # Set frozen modules to eval mode (keeps obs_encoder frozen)
        self.set_frozen_modules_to_eval_mode()

        gt_actions = action_input.action  # ground truth action [B, 16, action_dim]
        final_model_output_action = (
            action_head_output.final_model
        )  # [B, 16, hidden_dim]
        final_raw_action = action_head_output.final_raw_action  # [B, 16, action_dim]
        dt = action_head_output.dt
        observations = action_input.simple_img  # [B, V=1, T=16, H, W, C]

        # Run action_decoder_observe 16 times, each with corresponding obs and 1 action token
        pred_velocity_list = []
        for i in range(16):
            # Extract single observation frame from time dimension
            # observations shape: [B, V=1, T=16, H, W, C]
            obs_i = observations[:, :, i : i + 1, ...]  # [B, V=1, 1, H, W, C]

            # Slice corresponding 1 action token
            action_tokens_i = final_model_output_action[
                :, i : i + 1, :
            ]  # [B, 1, hidden_dim]

            # Get velocity prediction for this step (verbose only on first iteration)
            pred_vel_i = self.cross_modal_fusion_block(
                action_tokens_i, obs_i, action_input.embodiment_id, verbose=(i == 0)
            )  # [B, 1, action_dim]
            pred_velocity_list.append(pred_vel_i)

        # Concatenate all 16 velocity predictions to get [B, 16, action_dim]
        pred_velocity_all = torch.cat(pred_velocity_list, dim=1)  # [B, 16, action_dim]

        pred_actions = final_raw_action + dt * pred_velocity_all

        action_mask = action_input.action_mask
        loss = (
            F.mse_loss(pred_actions, gt_actions, reduction="none")
            * action_mask
        )
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    def get_action(
        self,
        action_head_output: BatchFeature,
        inference_idx: int,
        action_input: BatchFeature,
    ) -> BatchFeature:
        """
        Inference: Run feedback decoder once with current observation to get 1 action.

        At inference, simple_img has only 1 observation (T=1) - the current one.
        Each call predicts 1 action for the current timestep.

        Args:
            action_head_output: Cached outputs (final_model [B,16,hidden], final_raw_action [B,16,action_dim], dt)
            inference_idx: Index within the 16-action chunk (0-15)
            action_input: Contains simple_img [B, V=1, T=1, H, W, C] - single current observation

        Returns:
            1 action for the current timestep
        """
        # Set frozen modules to eval mode (keeps obs_encoder frozen)
        self.set_frozen_modules_to_eval_mode()

        action_idx = inference_idx % 16

        final_model_output_action = (
            action_head_output.final_model
        )  # [B, 16, hidden_dim]
        final_raw_action = action_head_output.final_raw_action  # [B, 16, action_dim]
        dt = action_head_output.dt
        observations = (
            action_input.simple_img
        )  # [B, V=1, T=1, H, W, C] - only 1 frame at inference

        # Slice the 1 action token for the current timestep
        action_token = final_model_output_action[
            :, action_idx : action_idx + 1, :
        ]  # [B, 1, hidden_dim]

        # Predict velocity from the current observation (single frame) using visual feedback decoder
        pred_vel = self.cross_modal_fusion_block(
            action_token,
            observations,  # keep as [B, V, T, H, W, C]; decoder will extract [B,H,W,C]
            action_input.embodiment_id,
            verbose=False,
        )  # [B, 1, action_dim]

        # Calculate predicted action for the current timestep
        raw_action = final_raw_action[
            :, action_idx : action_idx + 1, :
        ]  # [B, 1, action_dim]
        # pred_action = torch.tanh(raw_action + dt * pred_vel)  # [B, 1, action_dim]
        pred_action = raw_action + dt * pred_vel

        return BatchFeature(data={"action_pred": pred_action})

    @property
    def cross_modal_fusion_block(self):
        """LoopAct API; reuse the existing checkpoint storage key."""
        return self.action_decoder_observe

    # def get_action( # for debug use
    #     self,
    #     action_head_output: BatchFeature,
    #     inference_idx: int,
    #     action_input: BatchFeature,
    # ) -> BatchFeature:
    #     """
    #     Inference: Run feedback decoder once with current observation to get 4 actions.

    #     Unlike training where simple_img has 4 observations (T=4),
    #     at inference simple_img has only 1 observation (T=1) - the current one.

    #     Args:
    #         action_head_output: Cached outputs (final_model [B,16,hidden], final_raw_action [B,16,action_dim], dt)
    #         inference_idx: Index of inference call (increments every 4 env steps)
    #         action_input: Contains simple_img [B, V=1, T=1, H, W, C] - single current observation

    #     Returns:
    #         4 actions for the current window
    #     """
    #     # Set frozen modules to eval mode (keeps obs_encoder frozen)
    #     self.set_frozen_modules_to_eval_mode()

    #     window_idx = inference_idx % 4
    #     print(f"[VisualRefinementNetwork] inference_idx={inference_idx}, window_idx={window_idx}")

    #     baseline_action_pred = action_head_output.action_pred
    #     baseline_action_pred_window = baseline_action_pred[:, 0 * 4 : (0 + 1) * 4, :]

    #     return BatchFeature(data={"action_pred": baseline_action_pred_window})


# Serialization-compatible class aliases.
FeedbackDecoder = CrossModalFusionBlock
FeedbackAction = VisualRefinementNetwork
