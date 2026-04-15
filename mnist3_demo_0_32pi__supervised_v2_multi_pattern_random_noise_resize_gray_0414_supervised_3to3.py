"""
=============================================================================
Kuramoto 오실레이터 기반 Multi-Pattern Associative Memory
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from torchvision import datasets, transforms



# ============================
# 실험 파라미터
# ============================
PHASE_MIN = 1 * np.pi / 2
PHASE_MAX = 3 * np.pi / 2
GLOBAL_PHASE_SHIFT = 0


# ============================
# MNIST 로딩 + Resize
# ============================
def load_mnist(target_size=9):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    images = []
    labels = []

    for img, label in mnist:
        img = (img.squeeze().numpy() * 255).astype(np.uint8)

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_resized = torch.nn.functional.interpolate(
            img_tensor,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False
        )
        img_resized = img_resized.squeeze().numpy().astype(np.uint8)

        images.append(img_resized)
        labels.append(label)

    return np.array(images), np.array(labels)


# ============================
# MNIST → Phase encoding
# ============================
def mnist_to_phase(img, phase_min, phase_max):
    img = img.astype(np.float32)

    # [0, 255] -> [π/2, 3π/2]
    theta = (img / 255.0) * np.pi + (np.pi / 2)

    return theta.reshape(-1).astype(np.float32), img.shape


# ============================
# 이미지 마스킹
# ============================
def mask_top_right_quadrant(img):
    """
    오른쪽 위 사분면을 0으로 만든다.
    """
    img_masked = img.copy()
    H, W = img_masked.shape

    h_mid = H // 2
    w_mid = W // 2

    img_masked[:h_mid, w_mid:] = 0
    return img_masked


# ============================
# Gaussian corruption
# ============================
def add_gaussian_noise(img, noise_std=40.0, rng=None):
    """
    이미지 intensity domain에서 Gaussian noise 추가
    """
    if rng is None:
        rng = np.random.default_rng()

    img_f = img.astype(np.float32)
    noise = rng.normal(loc=0.0, scale=noise_std, size=img.shape).astype(np.float32)
    img_noisy = np.clip(img_f + noise, 0, 255).astype(np.uint8)
    return img_noisy


# ============================
# Salt-and-pepper corruption
# ============================
def add_salt_and_pepper_noise(img, salt_prob=0.1, pepper_prob=0.1, rng=None):
    """
    이미지에 salt-and-pepper noise 추가
    - salt: 255
    - pepper: 0
    """
    if rng is None:
        rng = np.random.default_rng()

    img_noisy = img.copy()
    H, W = img_noisy.shape

    rand_map = rng.random((H, W))

    img_noisy[rand_map < pepper_prob] = 0
    img_noisy[(rand_map >= pepper_prob) & (rand_map < pepper_prob + salt_prob)] = 255

    return img_noisy


# ============================
# Hebbian 초기화 (multi-pattern)
# ============================
def train_K_hebbian_multi(theta_list):
    N = theta_list[0].shape[0]
    K = np.zeros((N, N), dtype=np.float32)

    for theta in theta_list:
        delta = theta.reshape(N, 1) - theta.reshape(1, N)
        K += np.cos(delta)

    np.fill_diagonal(K, 0)
    K /= (N * len(theta_list))
    return K


# ============================
# bounded coupling
# ============================
def compute_K(W, K_max):
    return K_max * torch.tanh(W)


# ============================
# Kuramoto dynamics (train)
# ============================
def kuramoto_dynamics_train(theta0, W, K_max, steps=20, dt=0.02):
    theta = theta0

    for _ in range(steps):
        K = compute_K(W, K_max)
        K = K - torch.diag(torch.diagonal(K))

        delta = theta.unsqueeze(0) - theta.unsqueeze(1)
        dtheta = (K * torch.sin(delta)).sum(dim=1)

        theta = theta + dt * dtheta
        theta = torch.remainder(theta, 2 * np.pi)

    return theta


# ============================
# Kuramoto dynamics (inference)
# ============================
def kuramoto_dynamics(theta0, K, steps=20, dt=0.02):
    theta = torch.tensor(theta0, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)
    snaps = []

    for _ in range(steps):
        K2 = K - torch.diag(torch.diagonal(K))
        delta = theta.unsqueeze(0) - theta.unsqueeze(1)
        dtheta = (K2 * torch.sin(delta)).sum(dim=1)

        theta = theta + dt * dtheta
        theta = torch.remainder(theta, 2 * np.pi)

        snaps.append(theta.detach().numpy())

    return snaps


# ============================
# phase folding for visualization
# ============================
def fold_theta_to_display_range(theta):
    """
    theta를 [π/2, 3π/2] 구간으로 반사-fold 한다.
    - 먼저 [0, 2π) 로 wrap
    - [0, π/2) 구간은 π/2 경계에서 반사
    - (3π/2, 2π) 구간은 3π/2 경계에서 반사
    """
    theta_wrapped = np.mod(theta, 2 * np.pi)
    theta_folded = theta_wrapped.copy()

    low_mask = theta_wrapped < (np.pi / 2)
    high_mask = theta_wrapped > (3 * np.pi / 2)

    theta_folded[low_mask] = np.pi - theta_wrapped[low_mask]
    theta_folded[high_mask] = 3 * np.pi - theta_wrapped[high_mask]

    return theta_folded


# ============================
# torch용 fold + theta -> image
# ============================
def fold_theta_to_display_range_torch(theta):
    theta_wrapped = torch.remainder(theta, 2 * np.pi)
    theta_folded = theta_wrapped

    low_mask = theta_wrapped < (np.pi / 2)
    high_mask = theta_wrapped > (3 * np.pi / 2)

    theta_folded = torch.where(low_mask, np.pi - theta_wrapped, theta_folded)
    theta_folded = torch.where(high_mask, 3 * np.pi - theta_wrapped, theta_folded)

    return theta_folded


def theta_to_image_torch(theta):
    theta_folded = fold_theta_to_display_range_torch(theta)
    img = (theta_folded - (np.pi / 2)) / np.pi * 255.0
    img = torch.clamp(img, 0.0, 255.0)
    return img


# ============================
# pixel reconstruction loss
# ============================
def pixel_reconstruction_loss(theta_pred, img_target_flat):
    img_pred = theta_to_image_torch(theta_pred)
    return torch.mean((img_pred - img_target_flat) ** 2)


# ============================
# digit-specific corruption helper
# ============================
def make_corrupted_image_by_digit(digit, img, rng_gauss=None, rng_sp=None,
                                  gaussian_noise_std=40.0,
                                  sp_salt_prob=0.03,
                                  sp_pepper_prob=0.03):
    if digit == 0:
        img_corrupted = mask_top_right_quadrant(img)
        prefix = "digit_0_partial_loss"
    elif digit == 1:
        img_corrupted = add_gaussian_noise(
            img,
            noise_std=gaussian_noise_std,
            rng=rng_gauss
        )
        prefix = "digit_1_gaussian"
    elif digit == 3:
        img_corrupted = add_salt_and_pepper_noise(
            img,
            salt_prob=sp_salt_prob,
            pepper_prob=sp_pepper_prob,
            rng=rng_sp
        )
        prefix = "digit_3_saltpepper"
    else:
        img_corrupted = img.copy()
        prefix = f"digit_{digit}_clean"

    return img_corrupted, prefix


# ============================
# all-corruption helper for training/inference
# ============================
def make_corrupted_image_by_type(corruption_type, img, rng_gauss=None, rng_sp=None,
                                 gaussian_noise_std=40.0,
                                 sp_salt_prob=0.03,
                                 sp_pepper_prob=0.03):
    if corruption_type == "partial":
        img_corrupted = mask_top_right_quadrant(img)
    elif corruption_type == "gaussian":
        img_corrupted = add_gaussian_noise(
            img,
            noise_std=gaussian_noise_std,
            rng=rng_gauss
        )
    elif corruption_type == "saltpepper":
        img_corrupted = add_salt_and_pepper_noise(
            img,
            salt_prob=sp_salt_prob,
            pepper_prob=sp_pepper_prob,
            rng=rng_sp
        )
    else:
        raise ValueError(f"Unknown corruption_type: {corruption_type}")

    return img_corrupted


def make_corrupted_image_by_type_for_inference(corruption_type, digit, img,
                                               rng_gauss=None, rng_sp=None,
                                               gaussian_noise_std=40.0,
                                               sp_salt_prob=0.03,
                                               sp_pepper_prob=0.03):
    if corruption_type == "partial":
        img_corrupted = mask_top_right_quadrant(img)
        prefix = f"digit_{digit}_partial_loss"
    elif corruption_type == "gaussian":
        img_corrupted = add_gaussian_noise(
            img,
            noise_std=gaussian_noise_std,
            rng=rng_gauss
        )
        prefix = f"digit_{digit}_gaussian"
    elif corruption_type == "saltpepper":
        img_corrupted = add_salt_and_pepper_noise(
            img,
            salt_prob=sp_salt_prob,
            pepper_prob=sp_pepper_prob,
            rng=rng_sp
        )
        prefix = f"digit_{digit}_saltpepper"
    else:
        raise ValueError(f"Unknown corruption_type: {corruption_type}")

    return img_corrupted, prefix


# ============================
# 🔴 Visualization (fold 버전)
# ============================
def save_theta_image(theta, shape, path):
    H, W = shape
    theta_2d = theta.reshape(H, W)

    # 🔴 [0, 2π) -> fold to [π/2, 3π/2]
    theta_folded = fold_theta_to_display_range(theta_2d)

    # 🔴 [π/2, 3π/2] -> [0, 255]
    img = (theta_folded - (np.pi / 2)) / np.pi * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_uint8_image(img, path):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


# ============================
# Fixed-point export
# ============================

# theta → 10bit unsigned (0~1023)
def float_theta_to_u10(theta):
    q = np.round((theta % (2 * np.pi)) / (2 * np.pi) * 1023.0)
    q = np.clip(q, 0, 1023).astype(np.uint16)
    return q


# K → Q1.15 signed
def float_K_to_q15(K):
    q = np.round(K * 32768.0)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q


def export_theta_u10_carray(theta, name, path):
    q = float_theta_to_u10(theta.flatten())

    with open(path, "w") as f:
        f.write(f"unsigned char {name}[] = {{\n")
        for i, v in enumerate(q):
            lo = int(v) & 0xFF
            hi = (int(v) >> 8) & 0xFF
            f.write(f"   0x{lo:02X}, 0x{hi:02X},")
            if (i + 1) % 10 == 0:
                f.write(f"  // {i+1}\n")
        if len(q) % 10 != 0:
            f.write("\n")
        f.write("};\n")


def export_K_q15_carray(K, path, name="K_data"):
    q = float_K_to_q15(K.reshape(-1))

    with open(path, "w") as f:
        f.write(f"unsigned char {name}[] = {{\n")
        for i, v in enumerate(q):
            lo = int(v) & 0xFF
            hi = (int(v) >> 8) & 0xFF
            f.write(f"   0x{lo:02X}, 0x{hi:02X},")
            if (i + 1) % 10 == 0:
                f.write(f"  // {i+1}\n")
        if len(q) % 10 != 0:
            f.write("\n")
        f.write("};\n")


# ============================
# 저장 helper
# ============================
def save_inference_bundle(theta_init, theta_final, snaps, shape, out_dir, prefix):
    save_theta_image(theta_init, shape, os.path.join(out_dir, f"{prefix}_init.png"))

    for i, snap in enumerate(snaps):
        save_theta_image(
            snap,
            shape,
            os.path.join(out_dir, f"{prefix}_step_{i:02d}.png")
        )

    export_theta_u10_carray(
        theta_init,
        f"{prefix}_init",
        os.path.join(out_dir, f"{prefix}_init.txt")
    )
    export_theta_u10_carray(
        theta_final,
        f"{prefix}_final",
        os.path.join(out_dir, f"{prefix}_final.txt")
    )


# ============================
# main
# ============================
if __name__ == "__main__":


    digit_list = [3, 0, 7]
    index_list = [81, 35, 56]

    assert len(digit_list) == len(index_list)

    target_size = 8

    steps = 5
    dt = 0.25

    gaussian_noise_std = 40.0
    sp_salt_prob = 0.04
    sp_pepper_prob = 0.04

    train_iters = 3000
    lr = 3e-3
    K_max = 1.0
    lambda_phase = 0.05

    corruption_types_train = ["partial", "gaussian", "saltpepper"]
    corruption_types_infer = ["partial", "gaussian", "saltpepper"]

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("output", now)
    os.makedirs(out_dir, exist_ok=True)

    images, labels = load_mnist(target_size=target_size)

    theta_list_np = []
    theta_list_torch = []
    img_target_list_torch = []
    shape = None
    selected_imgs = {}

    print("[Dataset] loading patterns")
    print(f"[Resize] target_size = {target_size} x {target_size}")
    print(f"[Output] save dir = {os.path.abspath(out_dir)}")

    for digit, idx in zip(digit_list, index_list):
        img = images[np.where(labels == digit)[0][idx]]
        theta_np, shape = mnist_to_phase(img, PHASE_MIN, PHASE_MAX)

        selected_imgs[digit] = img
        theta_list_np.append(theta_np)
        theta_list_torch.append(torch.tensor(theta_np, dtype=torch.float32))
        img_target_list_torch.append(torch.tensor(img.reshape(-1), dtype=torch.float32))

        save_theta_image(theta_np, shape, f"{out_dir}/digit_{digit}_gt.png")
        save_uint8_image(img, f"{out_dir}/digit_{digit}_gt_uint8.png")
        export_theta_u10_carray(
            theta_np,
            f"theta_gt_digit_{digit}",
            os.path.join(out_dir, f"theta_gt_digit_{digit}.txt")
        )
        print(f"  digit {digit} loaded")

    N = theta_list_np[0].shape[0]
    print(f"[ONN] N = {N}, K shape = ({N}, {N})")

    print("[Stage 1] Hebbian multi-pattern pretraining")
    K_hebb = train_K_hebbian_multi(theta_list_np)
    np.save(os.path.join(out_dir, "K_hebbian.npy"), K_hebb)

    print("[Stage 2] Supervised fine-tuning for pixel reconstruction")
    print(f"[Training corruptions] {corruption_types_train}")

    eps = 1e-5
    K0 = np.clip(K_hebb / K_max, -1 + eps, 1 - eps)
    W_init = 0.5 * np.log((1 + K0) / (1 - K0))

    W = nn.Parameter(torch.tensor(W_init, dtype=torch.float32))
    optimizer = optim.Adam([W], lr=lr)

    best_loss = float("inf")
    best_W = None
    best_iter = -1

    for it in range(train_iters):
        optimizer.zero_grad()
        loss = 0.0

        rng_gauss_train = np.random.default_rng(1000 + it)
        rng_sp_train = np.random.default_rng(2000 + it)

        for digit, theta_target, img_target_flat in zip(digit_list, theta_list_torch, img_target_list_torch):
            img_clean = selected_imgs[digit]

            for corruption_type in corruption_types_train:
                img_corrupted = make_corrupted_image_by_type(
                    corruption_type,
                    img_clean,
                    rng_gauss=rng_gauss_train,
                    rng_sp=rng_sp_train,
                    gaussian_noise_std=gaussian_noise_std,
                    sp_salt_prob=sp_salt_prob,
                    sp_pepper_prob=sp_pepper_prob
                )

                theta_init_np, _ = mnist_to_phase(img_corrupted, PHASE_MIN, PHASE_MAX)
                theta_init = torch.tensor(theta_init_np, dtype=torch.float32)

                theta_pred = kuramoto_dynamics_train(theta_init, W, K_max, steps, dt)

                loss_pixel = pixel_reconstruction_loss(theta_pred, img_target_flat)
                loss_phase = torch.mean(1.0 - torch.cos(theta_pred - theta_target))
                loss = loss + loss_pixel + lambda_phase * loss_phase

        loss.backward()
        torch.nn.utils.clip_grad_norm_([W], 1.0)
        optimizer.step()

        with torch.no_grad():
            W.fill_diagonal_(0)

            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_W = W.detach().clone()
                best_iter = it

        if it % 20 == 0 or it == train_iters - 1:
            print(f"[Iter {it:04d}] loss = {loss.item():.6f}, best = {best_loss:.6f} @ iter {best_iter}")

    if best_W is None:
        raise RuntimeError("best_W was not saved during training.")

    print(f"[Best checkpoint] iter = {best_iter}, loss = {best_loss:.6f}")

    with torch.no_grad():
        W.copy_(best_W)
        K_trained = compute_K(W, K_max)
        K_trained = K_trained - torch.diag(torch.diagonal(K_trained))

    K_trained = K_trained.cpu().numpy()
    np.save(os.path.join(out_dir, "K_trained.npy"), K_trained)
    export_K_q15_carray(K_trained, os.path.join(out_dir, "K_data.txt"))

    print("[Inference] all digits x all corruptions")
    print(f"[Inference corruptions] {corruption_types_infer}")

    rng_gauss = np.random.default_rng(6)
    rng_sp = np.random.default_rng(20)

    for digit in digit_list:
        img = selected_imgs[digit]

        for corruption_type in corruption_types_infer:
            img_corrupted, prefix = make_corrupted_image_by_type_for_inference(
                corruption_type,
                digit,
                img,
                rng_gauss=rng_gauss,
                rng_sp=rng_sp,
                gaussian_noise_std=gaussian_noise_std,
                sp_salt_prob=sp_salt_prob,
                sp_pepper_prob=sp_pepper_prob
            )

            theta_init, _ = mnist_to_phase(img_corrupted, PHASE_MIN, PHASE_MAX)

            save_uint8_image(
                img_corrupted,
                os.path.join(out_dir, f"{prefix}_corrupted_uint8.png")
            )
            save_theta_image(
                theta_init,
                shape,
                os.path.join(out_dir, f"{prefix}_corrupted.png")
            )

            snaps = kuramoto_dynamics(theta_init, K_trained, steps, dt)
            theta_final = snaps[-1]

            save_inference_bundle(theta_init, theta_final, snaps, shape, out_dir, prefix)

    print("[Done] Hebbian + supervised pixel-reconstruction training complete.")