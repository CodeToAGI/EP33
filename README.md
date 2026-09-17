# EP33 — Transformer Architecture Explained  
**CodeToAGI Deep Learning Series · Module 8 · Episode 33**

> “Attention Is All You Need” — Vaswani et al., 2017  
> Self-Attention · Multi-Head · Positional Encoding · Encoder · Decoder

---

## What you will learn

- Why Transformers replaced RNNs (sequential bottleneck, vanishing gradients, path length)
- Self-attention: every token attends to every other token
- Q, K, V projections and why there are three of them
- Scaled dot-product attention formula derived step-by-step
- Multi-head attention (8 heads in parallel)
- Positional encoding (sine/cosine + learned embeddings)
- Full Encoder block: Multi-Head Self-Attn → Add & Norm → FFN → Add & Norm
- Full Decoder block: Masked Self-Attn → Cross-Attn → FFN
- Complete Encoder-Decoder Transformer diagram
- Clean PyTorch implementation of a Transformer Encoder from scratch

---

## Challenge

Implement a **2-layer Transformer Encoder** and verify the shape.

```bash
pip install torch
python ep33_transformer.py
Expected output:
textInput shape : torch.Size([8, 20])
Output shape: torch.Size([8, 20, 128])
Trainable parameters: ...
Post your output shape + parameter count in the comments or open an issue!

Key Formulas
textAttention(Q, K, V) = softmax(QKᵀ / √d_k) · V

MultiHead(Q, K, V) = Concat(head₁, …, headₕ) · W_O

PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

LayerNorm(x)  = (x − μ) / (σ + ε) · γ + β

Next Episodes

EP34 → BERT — Bidirectional Pretraining (Masked LM + NSP)
EP35 → GPT Family — Decoder-only Transformers at scale


Presenter: Mahaz Abbasi · AI Engineer

Channel: CodeToAGI
