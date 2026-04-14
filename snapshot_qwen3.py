import gc, torch
from collections import defaultdict
from transformers import AutoConfig, Qwen3Model

cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
cfg.use_cache = False
cfg.torch_dtype = torch.bfloat16
cfg._attn_implementation = "sdpa"

model = Qwen3Model(cfg).to(dtype=torch.bfloat16, device="cuda")
model.train()
ids = torch.randint(0, cfg.vocab_size, (1, 2048), device="cuda")

for _ in range(2):
    model.zero_grad(set_to_none=True)
    o = model(input_ids=ids)
    o.last_hidden_state.sum().backward()
    del o

gc.collect()
torch.cuda.empty_cache()
saved = []

def pack(t):
    saved.append({"shape": tuple(t.shape), "dtype": str(t.dtype),
                  "bytes": t.nelement() * t.element_size(), "ptr": t.data_ptr()})
    return t

model.zero_grad(set_to_none=True)
gc.collect()
torch.cuda.empty_cache()

with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    o = model(input_ids=ids)
    o.last_hidden_state.sum()

u = {t["ptr"]: t for t in saved}
g = defaultdict(list)
for t in u.values():
    g[round(t["bytes"] / 1024**2, 1)].append(t)

print(f"Unique tensors: {len(u)}")
for m in sorted(g, reverse=True):
    ts = g[m]
    c = len(ts)
    s = set(str(t["shape"]) for t in ts)
    print(f"{m:>8.1f} MiB x{c:>4}  {c/36:.1f}/layer  {list(s)[:4]}  {ts[0]['dtype']}")
