"""Runs ON THE GPU VM. Trains a small CIFAR-10 CNN with one config and prints one JSON line.

The switches are tricks from the public CIFAR-10 speedrun record (airbench, tysam hlb-CIFAR10):
- flip and translate augmentation;
- label smoothing;
- test-time augmentation;
- first-layer patch whitening;
- weight EMA;
- learning-rate schedules;
- width.

Truth comes from re-measurement: we never rely on the papers' numbers.

  python3 vm_train.py '<config json>' <seed> noisy   -> train on 45k, score on 5k held-out val (1 run)
  python3 vm_train.py '<config json>' <seed> truth   -> mean over seeds seed..seed+2, full 50k train, 10k test
"""
import json, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

torch.backends.cudnn.benchmark = True
dev = "cuda"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load(train):
    ds = torchvision.datasets.CIFAR10("/home/ubuntu/data", train=train, download=True)
    x = torch.tensor(ds.data).permute(0, 3, 1, 2).float().div(255)
    return ((x - MEAN) / STD), torch.tensor(ds.targets)


def make_model(width, whiten_w=None):
    w1, w2, w3 = {"small": (32, 64, 128), "medium": (64, 128, 256), "large": (96, 192, 384)}[width]

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.MaxPool2d(2), nn.BatchNorm2d(cout), nn.GELU(),
                             nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU())
    first = nn.Conv2d(3, 24, 2, padding=0, bias=True)
    if whiten_w is not None:
        first.weight.data.copy_(whiten_w); first.weight.requires_grad_(False)
    return nn.Sequential(first, nn.GELU(), block(24, w1), block(w1, w2), block(w2, w3),
                         nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(w3, 10, bias=False)).to(dev).to(memory_format=torch.channels_last)


def whitening_weights(x):
    # 2x2 patches -> eigen-decomposition -> 12 eigvecs scaled, plus negatives = 24 filters (airbench-style)
    p = x[:5000].unfold(2, 2, 1).unfold(3, 2, 1).permute(0, 2, 3, 1, 4, 5).reshape(-1, 12).double()
    cov = torch.cov(p.T)
    evals, evecs = torch.linalg.eigh(cov)
    W = (evecs.T / torch.sqrt(evals.clamp_min(1e-2)).unsqueeze(1)).float().reshape(12, 3, 2, 2)
    return torch.cat([W, -W], 0)


def augment(x, cfg, g):
    if cfg["flip_aug"]:
        m = torch.rand(len(x), device=dev, generator=g) < 0.5
        x = torch.where(m.view(-1, 1, 1, 1), x.flip(3), x)
    t = int(cfg["translate_aug"])
    if t > 0:
        xp = F.pad(x, (t, t, t, t), mode="reflect")
        dx = torch.randint(0, 2 * t + 1, (1,), device=dev, generator=g).item()
        dy = torch.randint(0, 2 * t + 1, (1,), device=dev, generator=g).item()
        x = xp[:, :, dy:dy + 32, dx:dx + 32]
    return x


def evaluate(model, x, y, tta):
    model.eval(); correct = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(x), 2000):
            xb = x[i:i + 2000].to(memory_format=torch.channels_last)
            logits = model(xb).float()
            if tta in ("flip", "flip_translate"):
                logits = logits + model(xb.flip(3)).float()
            if tta == "flip_translate":
                for dx, dy in ((1, 0), (0, 1)):
                    xs = F.pad(xb, (1, 1, 1, 1), mode="reflect")[:, :, dy:dy + 32, dx:dx + 32]
                    logits = logits + 0.5 * model(xs).float()
            correct += (logits.argmax(1) == y[i:i + 2000]).sum().item()
    return correct / len(x)


def train_once(cfg, seed, xtr, ytr, xte, yte):
    torch.manual_seed(seed); g = torch.Generator(device=dev); g.manual_seed(seed)
    ww = whitening_weights(xtr) if cfg["whiten_init"] else None
    model = make_model(cfg["width"], ww)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"])
    bs = int(cfg["batch_size"]); steps_per = len(xtr) // bs; total = steps_per * int(cfg["epochs"])
    sched = cfg["lr_schedule"]
    ema = None
    if cfg["ema"]:
        ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99))
    step = 0; t0 = time.time()
    for ep in range(int(cfg["epochs"])):
        model.train()
        perm = torch.randperm(len(xtr), device=dev, generator=g)
        for i in range(steps_per):
            idx = perm[i * bs:(i + 1) * bs]
            xb = augment(xtr[idx], cfg, g).to(memory_format=torch.channels_last); yb = ytr[idx]
            frac = step / max(total, 1)
            if sched == "linear_decay": f = 1 - frac
            elif sched == "onecycle": f = frac / 0.2 if frac < 0.2 else (1 - frac) / 0.8
            else: f = 1.0
            for pg in opt.param_groups: pg["lr"] = cfg["lr"] * max(f, 0.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb), yb, label_smoothing=float(cfg["label_smoothing"]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            if ema is not None: ema.update_parameters(model)
    eval_model = model
    if ema is not None:
        torch.optim.swa_utils.update_bn([(xtr[j:j + 1000].to(memory_format=torch.channels_last),) for j in range(0, 10000, 1000)], ema)
        eval_model = ema
    torch.cuda.synchronize()
    return evaluate(eval_model, xte, yte, cfg["tta"]), time.time() - t0


def main():
    cfg = json.loads(sys.argv[1]); seed = int(sys.argv[2]); mode = sys.argv[3]
    x, y = load(True); xt, yt = load(False)
    x, y, xt, yt = x.to(dev), y.to(dev), xt.to(dev), yt.to(dev)
    if mode == "noisy":
        gen = torch.Generator().manual_seed(12345)
        perm = torch.randperm(50000, generator=gen).to(dev)
        tr, va = perm[:45000], perm[45000:]
        acc, secs = train_once(cfg, seed, x[tr], y[tr], x[va], y[va])
        print(json.dumps({"acc": acc, "secs": secs}))
    else:
        accs, secs = [], []
        for s in range(seed, seed + 3):
            a, t = train_once(cfg, s, x, y, xt, yt); accs.append(a); secs.append(t)
        print(json.dumps({"acc": sum(accs) / len(accs), "accs": accs, "secs": sum(secs) / len(secs)}))


if __name__ == "__main__":
    main()
