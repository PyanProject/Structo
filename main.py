import os 
import argparse
import logging
import csv
from datetime import datetime
import time
import shutil
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import trimesh
import clip
from skimage import measure
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Импорт функций для загрузки датасета
from dataset_loader import load_dataset  # файл dataset_loader.py должен быть в PYTHONPATH или в той же папке

# =============================================================================
# Dataset: TestVoxelDataset с поддержкой ограничения количества образцов
# =============================================================================
class TestVoxelDataset(Dataset):
    """
    Загружает .off файлы из указанной директории, преобразует их в воксельное представление
    и генерирует текстовый prompt на основе имени файла.
    """
    def __init__(self, root_dir, voxel_size=64, transform=None, max_samples=0):
        self.root_dir = root_dir
        self.voxel_size = voxel_size
        self.transform = transform
        self.files = self._gather_files()
        if max_samples > 0:
            self.files = self.files[:max_samples]
        logging.info(f"[Dataset] Найдено файлов для обработки: {len(self.files)}")

    def _gather_files(self):
        files = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.lower().endswith(".off"):
                    files.append(os.path.join(dirpath, filename))
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        logging.debug(f"[Dataset] Обработка файла: {file_path}")
        start_time = time.time()
        mesh = self._load_mesh(file_path)
        if mesh is None:
            logging.warning(f"[Dataset] Файл {file_path} пропущен (не удалось загрузить mesh).")
            return None, None

        voxel_obj = self._voxelize_mesh(mesh)
        if voxel_obj is None:
            logging.warning(f"[Dataset] Файл {file_path} пропущен (не удалось вокселизировать mesh).")
            return None, None

        voxels = voxel_obj.matrix.astype(np.float32)
        voxel_tensor = self._center_voxel_grid(voxels)
        elapsed = time.time() - start_time
        logging.debug(f"[Dataset] Файл {file_path} обработан за {elapsed:.2f} сек.")

        object_name = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ')
        prompt = "3d model of " + object_name

        logging.debug(f"[Dataset] Объект: {object_name}, время обработки: {elapsed:.2f} сек, размер: {voxel_tensor.shape}")
        return voxel_tensor, prompt

    def _load_mesh(self, file_path):
        try:
            mesh = trimesh.load(file_path, file_type='off', force='mesh', process=False)
        except Exception as e:
            try:
                mesh = trimesh.load(file_path, file_type='off', process=False)
            except Exception as e:
                logging.error(f"[Dataset] Ошибка при загрузке {file_path}: {e}")
                return None
        return mesh

    def _voxelize_mesh(self, mesh):
        pitch = mesh.bounding_box.extents.max() / self.voxel_size
        try:
            voxel_obj = mesh.voxelized(pitch)
        except Exception as e:
            try:
                mesh = mesh.convex_hull
                voxel_obj = mesh.voxelized(pitch)
            except Exception as e:
                logging.error(f"[Dataset] Ошибка при вокселизации: {e}")
                return None
        return voxel_obj

    def _center_voxel_grid(self, voxels):
        vs = self.voxel_size
        grid = np.zeros((vs, vs, vs), dtype=voxels.dtype)
        v_shape = voxels.shape
        slices_voxels, slices_grid = [], []
        for s in v_shape:
            if s <= vs:
                start_grid = (vs - s) // 2
                slices_voxels.append(slice(0, s))
                slices_grid.append(slice(start_grid, start_grid + s))
            else:
                start_vox = (s - vs) // 2
                slices_voxels.append(slice(start_vox, start_vox + vs))
                slices_grid.append(slice(0, vs))
        grid[slices_grid[0], slices_grid[1], slices_grid[2]] = voxels[slices_voxels[0], slices_voxels[1], slices_voxels[2]]
        return torch.tensor(grid.tolist(), dtype=torch.float32).unsqueeze(0)

# =============================================================================
# Модель VoxelEncoder с индивидуальными блоками conv
# =============================================================================
class VoxelEncoder(nn.Module):
    def __init__(self, latent_dim, voxel_size, hidden_channels=32, num_layers=4, use_batch_norm=False, dropout_rate=0.0):
        super(VoxelEncoder, self).__init__()
        self.num_layers = num_layers
        in_channels = 1
        current_size = voxel_size
        for i in range(1, num_layers+1):
            out_channels = hidden_channels * (2 ** (i - 1))
            setattr(self, f"conv{i}", nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            if use_batch_norm:
                setattr(self, f"bn{i}", nn.BatchNorm3d(out_channels))
            setattr(self, f"relu{i}", nn.ReLU(inplace=True))
            if dropout_rate > 0:
                setattr(self, f"drop{i}", nn.Dropout3d(dropout_rate))
            in_channels = out_channels
            current_size //= 2
        self.flatten_dim = in_channels * (current_size ** 3)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        batch_size = x.size(0)
        for i in range(1, self.num_layers+1):
            x = getattr(self, f"conv{i}")(x)
            if hasattr(self, f"bn{i}"):
                x = getattr(self, f"bn{i}")(x)
            x = getattr(self, f"relu{i}")(x)
            if hasattr(self, f"drop{i}"):
                x = getattr(self, f"drop{i}")(x)
        x = x.view(batch_size, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# =============================================================================
# Модель ConditionalVoxelDecoder с индивидуальными блоками deconv
# =============================================================================
class ConditionalVoxelDecoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, voxel_size, hidden_channels=32, num_layers=4, use_batch_norm=False, dropout_rate=0.0):
        super(ConditionalVoxelDecoder, self).__init__()
        self.num_layers = num_layers
        s = voxel_size // (2 ** num_layers)
        initial_channels = hidden_channels * (2 ** (num_layers - 1))
        self.s = s
        self.fc = nn.Linear(latent_dim + cond_dim, initial_channels * (s ** 3))
        in_channels = initial_channels
        for i in range(1, num_layers):
            out_channels = in_channels // 2
            setattr(self, f"deconv{i}", nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            if use_batch_norm:
                setattr(self, f"bn_deconv{i}", nn.BatchNorm3d(out_channels))
            setattr(self, f"relu_deconv{i}", nn.ReLU(inplace=True))
            if dropout_rate > 0:
                setattr(self, f"drop_deconv{i}", nn.Dropout3d(dropout_rate))
            in_channels = out_channels
        setattr(self, f"deconv{num_layers}", nn.ConvTranspose3d(in_channels, 1, kernel_size=4, stride=2, padding=1))

    def forward(self, z, cond):
        z_cond = torch.cat([z, cond], dim=1)
        x = self.fc(z_cond)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.s, self.s, self.s)
        for i in range(1, self.num_layers):
            x = getattr(self, f"deconv{i}")(x)
            if hasattr(self, f"bn_deconv{i}"):
                x = getattr(self, f"bn_deconv{i}")(x)
            x = getattr(self, f"relu_deconv{i}")(x)
            if hasattr(self, f"drop_deconv{i}"):
                x = getattr(self, f"drop_deconv{i}")(x)
        x = getattr(self, f"deconv{self.num_layers}")(x)
        return x

# =============================================================================
# Объединённая модель CVAE_Conditional
# =============================================================================
class CVAE_Conditional(nn.Module):
    def __init__(self, latent_dim, voxel_size, cond_dim,
                 hidden_channels=32, num_encoder_layers=4, num_decoder_layers=4,
                 use_batch_norm=False, dropout_rate=0.0):
        super(CVAE_Conditional, self).__init__()
        self.encoder = VoxelEncoder(latent_dim, voxel_size, hidden_channels, num_encoder_layers, use_batch_norm, dropout_rate)
        self.decoder = ConditionalVoxelDecoder(latent_dim, cond_dim, voxel_size, hidden_channels, num_decoder_layers, use_batch_norm, dropout_rate)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, cond)
        return recon, mu, logvar

# =============================================================================
# Модель Discriminator
# =============================================================================
class Discriminator(nn.Module):
    def __init__(self, voxel_size, hidden_channels=32, num_layers=4, use_batch_norm=False, dropout_rate=0.0):
        super(Discriminator, self).__init__()
        layers = []
        in_channels = 1
        current_size = voxel_size
        for i in range(num_layers):
            out_channels = hidden_channels * (2 ** i)
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout3d(dropout_rate))
            in_channels = out_channels
            current_size //= 2
        self.conv = nn.Sequential(*layers)
        self.flatten_dim = in_channels * (current_size ** 3)
        self.fc = nn.Linear(self.flatten_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

# =============================================================================
# Utility Functions
# =============================================================================
def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy_with_logits(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar.float() - mu.float().pow(2) - logvar.float().exp())
    return recon_loss + kl_loss

def encode_prompt(prompt, device):
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model_clip = model_clip.float()
    tokens = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        prompt_embedding = model_clip.encode_text(tokens)
    return prompt_embedding

def compute_iou(recon, target, threshold=0.5):
    binary_recon = (recon > threshold).float()
    binary_target = (target > threshold).float()
    intersection = (binary_recon * binary_target).view(binary_recon.size(0), -1).sum(dim=1)
    union = ((binary_recon + binary_target) > 0).float().view(binary_recon.size(0), -1).sum(dim=1)
    return (intersection / (union + 1e-6)).mean()

def collate_fn_skip_none(batch):
    filtered = [item for item in batch if item[0] is not None]
    if len(filtered) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(filtered)

# =============================================================================
# Функция настройки логирования и создания директорий для сохранения результатов
# DEBUG-сообщения пишутся только в файл, а в консоль выводятся только INFO и выше.
# =============================================================================
def setup_logging_and_dirs(base_dir="rez"):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, f"run_{current_time}")
    os.makedirs(run_dir, exist_ok=True)
    weights_dir = os.path.join(run_dir, "weights")
    logs_dir = os.path.join(run_dir, "logs")
    for classification in ["good", "neutral", "bad"]:
        os.makedirs(os.path.join(weights_dir, classification), exist_ok=True)
        os.makedirs(os.path.join(logs_dir, classification), exist_ok=True)
    log_file = os.path.join(run_dir, "training.log")
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    
    # Удаляем предыдущие обработчики, если они есть
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
    logging.info(f"Run directory: {run_dir}")
    return run_dir, weights_dir, logs_dir

# =============================================================================
# Training Procedure
# =============================================================================
def train(args, device, run_dir, weights_dir, logs_dir):
    if args.parallel and args.num_workers == 0:
        args.num_workers = 4

    if args.device == "cpu":
        device = torch.device("cpu")
        logging.info("Selected device: CPU")
        try:
            import psutil
            mem = psutil.virtual_memory()
            logging.info(f"System RAM: Total: {mem.total/(1024**3):.2f} GB, Available: {mem.available/(1024**3):.2f} GB")
        except ImportError:
            logging.info("psutil не установлен, информация о RAM не доступна.")
    elif args.device == "gpu":
        if not torch.cuda.is_available():
            logging.error("GPU requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
            props = torch.cuda.get_device_properties(device)
            logging.info(f"Selected device: GPU ({props.name})")
            logging.info(f"Total VRAM: {props.total_memory/(1024**3):.2f} GB")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            props = torch.cuda.get_device_properties(device)
            logging.info(f"Auto-selected device: GPU ({props.name})")
            logging.info(f"Total VRAM: {props.total_memory/(1024**3):.2f} GB")
        else:
            device = torch.device("cpu")
            logging.info("Auto-selected device: CPU")
            try:
                import psutil
                mem = psutil.virtual_memory()
                logging.info(f"System RAM: Total: {mem.total/(1024**3):.2f} GB, Available: {mem.available/(1024**3):.2f} GB")
            except ImportError:
                logging.info("psutil не установлен, информация о RAM не доступна.")

    params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_params.txt")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Training parameters saved to {params_path}")

    dataset_dirs = load_dataset(args.dataset)
    if not dataset_dirs:
        raise ValueError("Dataset is empty. Check your dataset directory or known dataset name.")
    dataset_root = dataset_dirs[0]
    logging.info(f"Using dataset directory: {dataset_root}")
    dataset = TestVoxelDataset(root_dir=dataset_root, voxel_size=args.voxel_size, max_samples=args.max_samples)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your dataset directory.")
    
    num_files = len(dataset)
    sample_files = dataset.files[:10]
    logging.info(f"[Dataset] Всего файлов: {num_files}. Примеры: {sample_files}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn_skip_none)
    
    latent_dim = args.latent_dim
    cond_dim = args.cond_dim
    voxel_size = args.voxel_size
    cvae = CVAE_Conditional(latent_dim, voxel_size, cond_dim,
            hidden_channels=args.hidden_channels,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            use_batch_norm=args.use_batch_norm,
            dropout_rate=args.dropout_rate).to(device)
    
    discriminator = Discriminator(voxel_size,
            hidden_channels=args.disc_hidden_channels,
            num_layers=args.disc_num_layers,
            use_batch_norm=args.use_batch_norm,
            dropout_rate=args.dropout_rate).to(device)
    
    optimizer_G = optim.Adam(cvae.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr * 0.1)
    
    cvae.train()
    discriminator.train()
    
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model_clip.eval()

    checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
    
    use_amp = args.amp and torch.cuda.is_available()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        logging.info("Mixed precision training enabled (AMP).")
    else:
        logging.info("Mixed precision training disabled.")
    
    GOOD_IOU_THRESHOLD = 0.7
    BAD_IOU_THRESHOLD = 0.5

    logging.info(f"Starting training: epochs={args.epochs}, lr={args.lr}")
    logging.info(f"Model parameters: latent_dim={latent_dim}, cond_dim={cond_dim}, voxel_size={voxel_size}, hidden_channels={args.hidden_channels}, num_encoder_layers={args.num_encoder_layers}, num_decoder_layers={args.num_decoder_layers}, use_batch_norm={args.use_batch_norm}, dropout_rate={args.dropout_rate}")

    from tqdm import tqdm
    epoch_bar = tqdm(range(args.epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        epoch_loss = 0.0
        acc_real_total = 0.0
        acc_fake_total = 0.0
        acc_fake_forG_total = 0.0
        acc_iou_total = 0.0
        batch_count = 0

        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for voxels, prompts in batch_bar:
            if voxels is None:
                continue
            voxels = voxels.to(device)
            tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                cond = model_clip.encode_text(tokens)

            optimizer_D.zero_grad()
            real_labels = torch.ones(voxels.size(0), 1, device=device) * 0.9
            fake_labels = torch.zeros(voxels.size(0), 1, device=device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    recon, mu, logvar = cvae(voxels, cond)
                    D_real = discriminator(voxels)
                    loss_D_real = nn.functional.binary_cross_entropy_with_logits(D_real, real_labels)
                    D_fake = discriminator(recon.detach())
                    loss_D_fake = nn.functional.binary_cross_entropy_with_logits(D_fake, fake_labels)
                    loss_D = (loss_D_real + loss_D_fake) / 2
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                scaler.update()
            else:
                recon, mu, logvar = cvae(voxels, cond)
                D_real = discriminator(voxels)
                loss_D_real = nn.functional.binary_cross_entropy_with_logits(D_real, real_labels)
                D_fake = discriminator(recon.detach())
                loss_D_fake = nn.functional.binary_cross_entropy_with_logits(D_fake, fake_labels)
                loss_D = (loss_D_real + loss_D_fake) / 2
                loss_D.backward()
                optimizer_D.step()

            acc_real = (torch.sigmoid(D_real) >= 0.5).float().mean().item()
            acc_fake = (torch.sigmoid(D_fake) < 0.5).float().mean().item()

            total_gen_loss = 0.0
            for _ in range(2):
                optimizer_G.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast():
                        recon, mu, logvar = cvae(voxels, cond)
                        with torch.cuda.amp.autocast(enabled=False):
                            loss_vae_val = vae_loss(recon, voxels, mu, logvar)
                        D_fake_forG = discriminator(recon)
                        loss_adv = nn.functional.binary_cross_entropy_with_logits(D_fake_forG, real_labels)
                        warmup_factor = min((epoch + 1) / 5.0, 1.0)
                        loss_ratio = loss_adv.item() / (loss_vae_val.item() + 1e-8)
                        loss_ratio = max(0.1, min(loss_ratio, 10.0))
                        effective_lambda_adv = args.lambda_adv * warmup_factor * loss_ratio
                        loss_G_total = args.recon_weight * loss_vae_val + effective_lambda_adv * loss_adv
                    scaler.scale(loss_G_total).backward()
                    scaler.step(optimizer_G)
                    scaler.update()
                else:
                    recon, mu, logvar = cvae(voxels, cond)
                    loss_vae_val = vae_loss(recon, voxels, mu, logvar)
                    D_fake_forG = discriminator(recon)
                    loss_adv = nn.functional.binary_cross_entropy_with_logits(D_fake_forG, real_labels)
                    warmup_factor = min((epoch + 1) / 5.0, 1.0)
                    loss_ratio = loss_adv.item() / (loss_vae_val.item() + 1e-8)
                    loss_ratio = max(0.1, min(loss_ratio, 10.0))
                    effective_lambda_adv = args.lambda_adv * warmup_factor * loss_ratio
                    loss_G_total = args.recon_weight * loss_vae_val + effective_lambda_adv * loss_adv
                    loss_G_total.backward()
                    optimizer_G.step()
                total_gen_loss += loss_G_total.item()

            loss_G_total = total_gen_loss / 2.0
            acc_fake_forG = (torch.sigmoid(D_fake_forG) >= 0.5).float().mean().item()
            iou = compute_iou(recon, voxels)

            acc_real_total += acc_real
            acc_fake_total += acc_fake
            acc_fake_forG_total += acc_fake_forG
            acc_iou_total += iou.item()
            batch_count += 1
            epoch_loss += loss_G_total

            batch_bar.set_postfix(loss=loss_G_total, D_real=acc_real, D_fake=acc_fake)

        avg_loss = epoch_loss / batch_count
        avg_acc_real = acc_real_total / batch_count
        avg_acc_fake = acc_fake_total / batch_count
        avg_acc_gen = acc_fake_forG_total / batch_count
        avg_iou = acc_iou_total / batch_count

        epoch_info = (
            f"Epoch [{epoch+1}/{args.epochs}]: Loss={avg_loss:.4f}, "
            f"D_Acc(Real)={avg_acc_real:.4f}, D_Acc(Fake)={avg_acc_fake:.4f}, "
            f"G_Acc={avg_acc_gen:.4f}, IoU={avg_iou:.4f}"
        )
        logging.info(epoch_info)
        epoch_bar.set_postfix_str(epoch_info)

        if avg_iou >= GOOD_IOU_THRESHOLD:
            classification = "good"
        elif avg_iou <= BAD_IOU_THRESHOLD:
            classification = "bad"
        else:
            classification = "neutral"
        classification_line = f"{datetime.now()} - Epoch {epoch+1} classified as: {classification} | {epoch_info}\n"
        logging.info(f"Epoch {epoch+1} classified as: {classification}")
        class_log_path = os.path.join(logs_dir, classification, "training_epoch.log")
        with open(class_log_path, "a", encoding="utf-8") as clf:
            clf.write(classification_line)

        torch.save({
            "epoch": epoch + 1,
            "cvae_state_dict": cvae.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
        }, checkpoint_path)
        logging.info(f"Checkpoint saved (overwritten) at: {checkpoint_path}")

    logging.info("Training completed successfully.")

    # Принудительно сбрасываем буферы логгера и закрываем его,
    # чтобы все сообщения были записаны вне зависимости от результата.
    for handler in logging.getLogger().handlers:
        handler.flush()
    logging.shutdown()

    final_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_folder = os.path.join("rez", classification, final_time)
    os.makedirs(final_folder, exist_ok=True)
    log_file_src = os.path.join(run_dir, "training.log")
    log_file_dst = os.path.join(final_folder, "training.log")
    shutil.copy(log_file_src, log_file_dst)
    checkpoint_dst = os.path.join(final_folder, "checkpoint.pth")
    shutil.copy(checkpoint_path, checkpoint_dst)
    metrics_file = os.path.join(final_folder, "metrics.txt")
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(epoch_info + "\n")
    logging.info(f"Final logs and model saved to: {final_folder}")

# =============================================================================
# Generation Procedure
# =============================================================================
def generate(args, device):
    logging.info("Starting 3D model generation")
    latent_dim = args.latent_dim
    cond_dim = args.cond_dim
    cvae = CVAE_Conditional(latent_dim, args.voxel_size, cond_dim,
             hidden_channels=args.hidden_channels,
             num_encoder_layers=args.num_encoder_layers,
             num_decoder_layers=args.num_decoder_layers,
             use_batch_norm=args.use_batch_norm,
             dropout_rate=args.dropout_rate).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cvae.load_state_dict(checkpoint["cvae_state_dict"])
    cvae.eval()

    z = torch.randn(1, latent_dim).to(device)
    if args.prompt is not None:
        cond = encode_prompt(args.prompt, device)
    else:
        cond = torch.zeros(1, cond_dim).to(device)
    with torch.no_grad():
        voxel_out = cvae.decoder(z, cond)
    voxel_grid = voxel_out.squeeze().cpu().numpy()
    voxel_grid = gaussian_filter(voxel_grid, sigma=1)

    v_min, v_max = voxel_grid.min(), voxel_grid.max()
    if not (v_min <= 0.5 <= v_max):
        level = (v_min + v_max) / 2
        logging.warning(f"Level 0.5 not in range [{v_min:.3f}, {v_max:.3f}]. Using level {level:.3f}.")
    else:
        level = 0.5

    verts, faces, normals, values = measure.marching_cubes(voxel_grid, level=level)
    with open(args.output, "w") as f:
        for v in verts:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            f.write("f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1))
    logging.info(f"3D model generated and saved to {args.output}")

    return args.output

# =============================================================================
# Analysis Procedure for CLIP Embeddings
# =============================================================================
def analyze_clip_embeddings(args, device):
    logging.info("Starting CLIP embeddings analysis")
    model, _ = clip.load("ViT-B/32", device=device)
    texts = [
        "3d model of chair",
        "3d model of airplane",
        "3d model of table",
        "3d model of car",
        "3d model of sofa"
    ]
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
    text_embeddings = text_embeddings.cpu().numpy()
    normed_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    cosine_sim = np.dot(normed_embeddings, normed_embeddings.T)
    logging.info("Cosine similarity matrix:")
    logging.info(cosine_sim)

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(text_embeddings)
    plt.figure(figsize=(8, 6))
    for i, text in enumerate(texts):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label=text)
        plt.text(reduced_embeddings[i, 0] + 0.01, reduced_embeddings[i, 1] + 0.01, text)
    plt.title("PCA Projection of CLIP Text Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()
    logging.info("CLIP embeddings analysis completed.")

# =============================================================================
# Main Function with Argument Parsing
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="VAE-GAN for text-to-3D Model Generation on custom dataset"
    )
    parser.add_argument("--mode", type=str, choices=["train", "generate", "analyze"], required=True,
                        help="Mode: train, generate or analyze")
    parser.add_argument("--dataset", type=str, default="testdataset",
                        help="Path or known name of dataset (e.g. 'chairs')")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for generator")
    parser.add_argument("--checkpoint", type=str, default="vae_gan_test.pth", help="Checkpoint file path")
    parser.add_argument("--voxel_size", type=int, default=64, help="Voxel grid resolution")
    parser.add_argument("--output", type=str, default="model.obj", help="Output OBJ file for generation")
    parser.add_argument("--lambda_adv", type=float, default=0.001, help="Weight factor for adversarial loss")
    parser.add_argument("--recon_weight", type=float, default=10.0, help="Weight factor for reconstruction loss")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for conditional 3D generation")
    parser.add_argument("--cond_dim", type=int, default=512, help="Dimension of the condition vector")
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimension of latent vector")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit number of samples (0 for no limit)")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision training")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "gpu"], default="auto",
                        help="Device to use: auto, cpu, or gpu")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    # Новые параметры для архитектуры
    parser.add_argument("--hidden_channels", type=int, default=32, help="Initial number of hidden channels for encoder/decoder")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Number of layers in the encoder")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="Number of layers in the decoder")
    parser.add_argument("--use_batch_norm", action="store_true", help="Use batch normalization in the networks")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate (0.0 means no dropout)")
    parser.add_argument("--disc_hidden_channels", type=int, default=32, help="Initial hidden channels for discriminator")
    parser.add_argument("--disc_num_layers", type=int, default=4, help="Number of layers in the discriminator")
    
    args = parser.parse_args()
    args.checkpoint = "vae_gan_test.pth"

    # Создаем единую директорию для логирования и результатов
    run_dir, weights_dir, logs_dir = setup_logging_and_dirs()
    logging.info("Logging is configured and test message logged.")

    device = None

    if args.mode == "train":
        train(args, device, run_dir, weights_dir, logs_dir)
    elif args.mode == "generate":
        generate(args, device)
    elif args.mode == "analyze":
        analyze_clip_embeddings(args, device)

if __name__ == "__main__":
    main()
