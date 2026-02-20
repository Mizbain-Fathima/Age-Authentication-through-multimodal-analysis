import torch

ckpt = torch.load("models/fusion_age_model_best.pth", map_location="cpu")

print(ckpt.keys())
print("Best val MAE:", ckpt.get("best_val_mae"))
print("Epoch:", ckpt.get("epoch"))

print("Best val loss:", ckpt.get("best_val_loss"))
print("Epoch:", ckpt.get("epoch") + 1)

history = ckpt["history"]

print(history.keys())
print("Val losses per epoch:", history.get("val_loss"))
print("Val MAEs per epoch:", history.get("val_mae"))
