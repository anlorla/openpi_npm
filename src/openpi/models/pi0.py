import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_goal_mask = config.use_goal_mask
        self.goal_mask_resolution = config.goal_mask_resolution
        self.goal_mask_channels = config.goal_mask_channels

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        # Goal mask encoder (small CNN) - Approach A
        if self.use_goal_mask:
            # Input channels: goal_mask + delta_goal_mask
            mask_input_channels = config.goal_mask_channels * 2
            # Simple CNN: Conv -> ReLU -> Conv -> ReLU -> GlobalAvgPool -> MLP
            self.mask_conv1 = nnx.Conv(
                mask_input_channels, 32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs
            )
            self.mask_conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
            self.mask_conv3 = nnx.Conv(64, 128, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
            # After 3 strides of 2, 64x64 -> 8x8
            # GlobalAvgPool will reduce 8x8 -> 1, so we have 128 features
            self.mask_mlp = nnx.Linear(128, config.goal_mask_latent_dim, rngs=rngs)

            # Fusion MLP: concatenate mask latent with action tokens, then project back
            # This will be applied in embed_suffix
            self.mask_fusion_mlp = nnx.Linear(
                action_expert_config.width + config.goal_mask_latent_dim,
                action_expert_config.width,
                rngs=rngs
            )

        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def encode_goal_mask(
        self, obs: _model.Observation
    ) -> at.Float[at.Array, "b latent_dim"] | None:
        """
        Encode goal_mask and delta_goal_mask into a latent vector z_k.

        Args:
            obs: Observation containing goal_mask and delta_goal_mask

        Returns:
            Latent vector z_k of shape [b, latent_dim], or None if goal_mask is not provided
        """
        if not self.use_goal_mask or obs.goal_mask is None:
            return None

        # Concatenate goal_mask and delta_goal_mask along channel dimension
        # goal_mask: [b, h, w, c], delta_goal_mask: [b, h, w, c]
        if obs.delta_goal_mask is not None:
            mask_input = jnp.concatenate([obs.goal_mask, obs.delta_goal_mask], axis=-1)  # [b, h, w, 2*c]
        else:
            # If delta_goal_mask is not provided, use zeros
            mask_input = jnp.concatenate([obs.goal_mask, jnp.zeros_like(obs.goal_mask)], axis=-1)

        # Downsample to goal_mask_resolution if necessary
        if mask_input.shape[1:3] != self.goal_mask_resolution:
            from openpi.shared import image_tools
            mask_input = image_tools.resize_with_pad(mask_input, *self.goal_mask_resolution)

        # CNN encoder: Conv layers with ReLU activations
        x = self.mask_conv1(mask_input)
        x = nnx.relu(x)
        x = self.mask_conv2(x)
        x = nnx.relu(x)
        x = self.mask_conv3(x)
        x = nnx.relu(x)

        # Global average pooling: [b, h, w, c] -> [b, c]
        x = jnp.mean(x, axis=(1, 2))

        # MLP to project to latent dimension
        z_k = self.mask_mlp(x)  # [b, latent_dim]

        return z_k

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        # Approach A: Fuse goal_mask latent with action tokens
        if self.use_goal_mask:
            z_k = self.encode_goal_mask(obs)  # [b, latent_dim]
            if z_k is not None:
                # Broadcast z_k to all action tokens: [b, latent_dim] -> [b, action_horizon, latent_dim]
                z_k_expanded = einops.repeat(z_k, "b d -> b s d", s=self.action_horizon)
                # Concatenate with action tokens: [b, ah, width] + [b, ah, latent_dim] -> [b, ah, width+latent_dim]
                fused = jnp.concatenate([action_expert_tokens, z_k_expanded], axis=-1)
                # Project back to width: [b, ah, width+latent_dim] -> [b, ah, width]
                action_expert_tokens = self.mask_fusion_mlp(fused)
                action_expert_tokens = nnx.swish(action_expert_tokens)

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _velocity(
        self,
        observation: _model.Observation,
        x_t: at.Float[at.Array, "b ah ad"],
        time: at.Float[at.Array, " b"],
        kv_cache,
    ) -> at.Float[at.Array, "b ah ad"]:
        """
        Compute velocity field v_π(x_t, observation, time).

        Extracted from sample_actions to be reused by realtime_action.
        This ensures both methods use identical velocity computation.

        Args:
            observation: Current observation (preprocessed)
            x_t: Noisy actions at timestep time [b, ah, ad]
            time: Diffusion timestep [b]
            kv_cache: Cached prefix KV states

        Returns:
            Velocity field v_t [b, ah, ad]
        """
        batch_size = x_t.shape[0]

        # Embed suffix (action + time)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        # Construct attention masks
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)

        # Get prefix mask for cross-attention
        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

        # Compute positions
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        # Forward through transformer
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None

        # Project to action space
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return v_t

    def realtime_action(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_flow_steps: int = 10,
        prev_action_chunk: at.Float[at.Array, "b ah ad"],
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 100.0,
    ) -> _model.Actions:
        """
        Real-time action generation with VJP-based guidance (Algorithm 1, GUIDEDINFERENCE).

        Implements guided diffusion sampling that matches prev_action_chunk in the prefix region
        using vector-Jacobian products (VJP) for implicit differentiation.

        Key steps (matching paper Algorithm 1):
        1. Compute prefix attention weights (soft mask) - Eq. 5
        2. Initialize from noise
        3. For each diffusion step:
           a. Define denoiser f(x) = x + (1-τ) * v_π(x, o, τ) - line 26
           b. Compute VJP to get gradient ∂f/∂x - line 27
           c. Calculate error in prefix region with soft mask - line 27
           d. Apply correction with adaptive guidance weight - line 28
           e. Update with corrected velocity - line 29

        Args:
            rng: Random key for noise initialization
            observation: Current observation
            num_flow_steps: Number of denoising steps (n in paper)
            prev_action_chunk: Previous action chunk A_prev [b, H, D]
            inference_delay: Committed prefix length (d in paper)
            prefix_attention_horizon: Where prefix attention ends (H - s in paper)
            prefix_attention_schedule: Weight decay schedule ("linear", "exp", "ones", "zeros")
            max_guidance_weight: Maximum guidance strength (β in paper)

        Returns:
            Generated action chunk [b, ah, ad] aligned with prev_action_chunk in prefix
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_flow_steps
        batch_size = observation.state.shape[0]

        # Initialize from noise (same as sample_actions)
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Setup KV cache with prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # Compute prefix attention weights (Eq. 5 in paper)
        weights = self._get_prefix_weights(
            inference_delay,
            prefix_attention_horizon,
            self.action_horizon,
            prefix_attention_schedule
        )  # [H]

        def step(carry):
            x_t, time = carry
            time_batch = jnp.broadcast_to(time, (batch_size,))

            # Define denoiser for VJP (Algorithm 1, line 26)
            # f(x) = x + (1-τ) * v_π(x, o, τ)
            def denoiser(x_t_input):
                v_t = self._velocity(observation, x_t_input, time_batch, kv_cache)
                x_1 = x_t_input + v_t * (1 - time)
                return x_1, v_t

            # Compute VJP for implicit differentiation (Algorithm 1, line 27)
            # This gives us ∂x_1/∂x_t, which tells us how to adjust x_t to match prev_action_chunk
            x_1_pred, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)

            # Prefix error with soft masking (Algorithm 1, line 27)
            # Only enforce matching in the prefix region (weighted by schedule)
            error = (prev_action_chunk - x_1_pred) * weights[None, :, None]  # [b, H, D]

            # VJP correction: gradient of error w.r.t. x_t (Algorithm 1, line 28)
            (pinv_correction,) = vjp_fun(error)

            # Adaptive guidance weight (Eq. 5 in paper)
            # Stronger correction when far from target (time close to 1)
            inv_r2 = (time**2 + (1 - time)**2) / ((1 - time)**2)
            c = jnp.nan_to_num((1 - time) / time, posinf=max_guidance_weight)
            guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)

            # Apply correction to velocity (Algorithm 1, line 29)
            v_t_corrected = v_t + guidance_weight * pinv_correction

            return x_t + dt * v_t_corrected, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _get_prefix_weights(
        self,
        start: int,
        end: int,
        total: int,
        schedule: str
    ) -> at.Float[at.Array, " {total}"]:
        """
        Compute prefix attention weights matching RTC paper Eq. 5.

        Generates soft mask weights that:
        - Force alignment in committed region [0, start)
        - Smoothly transition in attention region [start, end)
        - Allow free generation in [end, total)

        Args:
            start: inference_delay (d) - where free generation starts
            end: prefix_attention_horizon (H - s) - where prefix attention ends
            total: action_horizon (H)
            schedule: Weight decay schedule

        Returns:
            Weights [w_0, ..., w_{total-1}] where:
            - w_i = 1.0 for i < start (must match prev_chunk)
            - w_i decays from 1.0 to 0.0 for start <= i < end
            - w_i = 0.0 for i >= end (pure new generation)
        """
        start = jnp.minimum(start, end)
        indices = jnp.arange(total)

        if schedule == "ones":
            # Constant weight (for debugging)
            return jnp.ones(total)
        elif schedule == "zeros":
            # Hard cutoff at start (no transition)
            return (indices < start).astype(jnp.float32)
        elif schedule == "linear":
            # Linear decay in transition zone
            alpha = jnp.clip((start - 1 - indices) / (end - start + 1) + 1, 0, 1)
            return jnp.where(indices >= end, 0.0, alpha)
        elif schedule == "exp":
            # Exponential decay (paper default, smoother than linear)
            # Formula from paper: w = alpha * (exp(alpha) - 1) / (e - 1)
            alpha = jnp.clip((start - 1 - indices) / (end - start + 1) + 1, 0, 1)
            alpha = alpha * jnp.expm1(alpha) / (jnp.e - 1)
            return jnp.where(indices >= end, 0.0, alpha)
        else:
            raise ValueError(f"Unknown schedule: {schedule}. Must be 'ones', 'zeros', 'linear', or 'exp'.")
