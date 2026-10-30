# Locked to THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219
# scripts/models/qwen3.5-4B.sh.  Qwen3.5 uses ordinary MCore/Transformer
# Engine blocks for full attention and the plugin spec below for GDN blocks.
MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 4
   --kv-channels 256
   --num-layers 32
   --hidden-size 2560
   --ffn-hidden-size 9216
   --use-gated-attention
   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --vocab-size 248320
   --rotary-base 10000000
   # The immutable MCore 0.14 image exposes the same Qwen gate under
   # --use-gated-attention (linear_qgkv).  New MCore calls this
   # --attention-output-gate (linear_qkv); the tensor adapters support both.
)
