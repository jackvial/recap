uv run python -m lerobot.rl.algorithms.RECAPSmolVLAValueNetwork
RECAPSmolVLAValueNetwork(
  (smolvlm): SmolVLMForConditionalGeneration(
    (model): SmolVLMModel(
      (vision_model): SmolVLMVisionTransformer(
        (embeddings): SmolVLMVisionEmbeddings(
          (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), padding=valid)
          (position_embedding): Embedding(1024, 768)
        )
        (encoder): SmolVLMEncoder(
          (layers): ModuleList(
            (0-11): 12 x SmolVLMEncoderLayer(
              (self_attn): SmolVLMVisionAttention(
                (k_proj): Linear(in_features=768, out_features=768, bias=True)
                (v_proj): Linear(in_features=768, out_features=768, bias=True)
                (q_proj): Linear(in_features=768, out_features=768, bias=True)
                (out_proj): Linear(in_features=768, out_features=768, bias=True)
              )
              (layer_norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
              (mlp): SmolVLMVisionMLP(
                (activation_fn): GELUTanh()
                (fc1): Linear(in_features=768, out_features=3072, bias=True)
                (fc2): Linear(in_features=3072, out_features=768, bias=True)
              )
              (layer_norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            )
          )
        )
        (post_layernorm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
      )
      (connector): SmolVLMConnector(
        (modality_projection): SmolVLMSimpleMLP(
          (proj): Linear(in_features=12288, out_features=960, bias=False)
        )
      )
      (text_model): LlamaModel(
        (embed_tokens): Embedding(49280, 960, padding_idx=2)
        (layers): ModuleList(
          (0-15): 16 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): Linear(in_features=960, out_features=960, bias=False)
              (k_proj): Linear(in_features=960, out_features=320, bias=False)
              (v_proj): Linear(in_features=960, out_features=320, bias=False)
              (o_proj): Linear(in_features=960, out_features=960, bias=False)
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear(in_features=960, out_features=2560, bias=False)
              (up_proj): Linear(in_features=960, out_features=2560, bias=False)
              (down_proj): Linear(in_features=2560, out_features=960, bias=False)
              (act_fn): SiLUActivation()
            )
            (input_layernorm): LlamaRMSNorm((960,), eps=1e-05)
            (post_attention_layernorm): LlamaRMSNorm((960,), eps=1e-05)
          )
        )
        (norm): LlamaRMSNorm((960,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
      )
    )
    (lm_head): Linear(in_features=960, out_features=49280, bias=False)
  )
  (state_proj): Linear(in_features=32, out_features=960, bias=True)
  (fusion_head): Sequential(
    (0): Linear(in_features=960, out_features=960, bias=True)
    (1): SiLU()
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=960, out_features=960, bias=True)
    (4): SiLU()
  )
  (value_head): Linear(in_features=960, out_features=201, bias=True)
)
Total parameters: 352,235,145
Trainable parameters: 352,235,145