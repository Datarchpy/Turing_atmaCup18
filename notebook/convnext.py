# atmaCup 18 - ConvNeXt CNN Baseline
# カメラ情報をCNNで処理し、予測値とテーブルデータをLightGBMでstackingする手法のベースライン

# ========== ライブラリのインポート ==========
import json
import multiprocessing
import cv2
import albumentations as A
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import warnings
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import importlib
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
import datetime
import math
import random
from timm.utils.model_ema import ModelEmaV2

# ========== 設定クラス ==========
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CFG:
    # ============== コンペティション基本設定 =============
    comp_name = 'atmacup_18'  # コンペティション名
    
    # Kaggle環境のパス設定
    comp_dir_path = '/kaggle/input/'
    comp_folder_name = 'atmacup18-dataset/atmaCup18_dataset'  # データセットフォルダ名
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'

    exp_name = 'atmacup_18_convnext_exp001'  # 実験名

    is_debug = False  # デバッグモード（提出時はFalseに設定）
    use_gray_scale = False  # グレースケール使用フラグ

    model_in_chans = 9  # モデルの入力チャンネル数（3フレーム × 3チャンネル）

    # ============== ファイルパス設定 =============
    train_fold_dir = "./working/folds/"  # フォールドデータ保存パス
    train_csv_path = f"{comp_dataset_path}train_features.csv"
    test_csv_path = f"{comp_dataset_path}test_features.csv"

    # ============== モデル設定 =============
    model_name = 'convnext_tiny'  # 使用するモデル名

    num_frames = 3  # 使用するフレーム数
    norm_in_chans = 1 if use_gray_scale else 3  # 正規化用チャンネル数

    use_torch_compile = False  # PyTorch 2.0のコンパイル機能
    use_ema = False  # Exponential Moving Average使用フラグ
    ema_decay = 0.995  # EMAの減衰率

    # ============== 訓練設定 =============
    size = 224  # 画像サイズ
    batch_size = 32  # バッチサイズ
    use_amp = True  # 混合精度訓練使用フラグ

    scheduler = 'GradualWarmupSchedulerV2'  # 学習率スケジューラ
    epochs = 20  # エポック数
    if is_debug:
        epochs = 1  # デバッグ時は1エポック

    # 学習率設定（AdamW + warmup）
    warmup_factor = 10
    lr = 2e-4  # 基本学習率
    if scheduler == 'GradualWarmupSchedulerV2':
        lr /= warmup_factor  # warmup使用時は初期学習率を下げる

    # ============== クロスバリデーション設定 =============
    n_fold = 4  # フォールド数
    use_holdout = False  # ホールドアウト検証フラグ
    use_alldata = False  # 全データ使用フラグ
    train_folds = [0, 1, 2, 3, 4]  # 訓練に使用するフォールド

    skf_col = 'class'  # Stratified K-Fold用カラム
    group_col = 'scene'  # Group K-Fold用カラム（シーン別分割）
    fold_type = 'gkf'  # フォールド分割方法

    objective_cv = 'regression'  # 目的変数タイプ
    metric_direction = 'minimize'  # メトリック最適化方向
    metrics = 'calc_mae_atmacup'  # 評価指標

    # ============== 予測ターゲット設定 =============
    target_size = 18  # 予測ターゲット数
    # 6つの物体の3D座標 (x, y, z) × 6 = 18次元
    target_col = ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2',
                  'z_2', 'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4', 'x_5', 'y_5', 'z_5']

    # ============== その他設定 =============
    pretrained = True  # 事前訓練済みモデル使用
    inf_weight = 'best'  # 推論時に使用する重み

    min_lr = 1e-6  # 最小学習率
    weight_decay = 1e-6  # 重み減衰
    max_grad_norm = 10  # 勾配クリッピング閾値

    print_freq = 500  # ログ出力頻度
    num_workers = 4  # データローダーのワーカー数
    seed = 42  # 乱数シード

    # ============== パス設定 =============
    if exp_name is not None:
        print('set dataset path')
        
        outputs_path = f'/kaggle/working/{exp_name}/'
        submission_dir = outputs_path + 'submissions/'
        submission_path = submission_dir + f'submission_{exp_name}.csv'
        model_dir = outputs_path + f'{comp_name}-models/'
        figures_dir = outputs_path + 'figures/'
        log_dir = outputs_path + 'logs/'
        log_path = log_dir + f'{exp_name}.txt'

    # ============== データ拡張設定 =============
    # 訓練時のデータ拡張
    train_aug_list = [
        A.Resize(size, size),  # リサイズ
        A.OneOf([  # ノイズ・ブラー系拡張をランダムに適用
            A.GaussNoise(var_limit=[10, 50]),  # ガウシアンノイズ
            A.GaussianBlur(),  # ガウシアンブラー
            A.MotionBlur(),  # モーションブラー
        ], p=0.4),
        A.Normalize(  # 正規化
            mean=[0] * norm_in_chans*num_frames,
            std=[1] * norm_in_chans*num_frames, 
        ),
        ToTensorV2(),  # テンソル変換
    ]

    # 検証時のデータ拡張（正規化のみ）
    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean=[0] * norm_in_chans*num_frames,
            std=[1] * norm_in_chans*num_frames,
        ),
        ToTensorV2(),
    ]

# ========== フォールド分割関数 ==========
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, StratifiedGroupKFold

def get_fold(train, cfg):
    """データをフォールドに分割する関数"""
    if cfg.fold_type == 'kf':
        # K-Fold
        Fold = KFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
        kf = Fold.split(train, train[cfg.target_col])
    elif cfg.fold_type == 'skf':
        # Stratified K-Fold
        Fold = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
        kf = Fold.split(train, train[cfg.skf_col])
    elif cfg.fold_type == 'gkf':
        # Group K-Fold（シーン別に分割）
        Fold = GroupKFold(n_splits=cfg.n_fold)
        groups = train[cfg.group_col].values
        kf = Fold.split(train, train[cfg.group_col], groups)
    elif cfg.fold_type == 'sgkf':
        # Stratified Group K-Fold
        Fold = StratifiedGroupKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
        groups = train[cfg.group_col].values
        kf = Fold.split(train, train[cfg.skf_col], groups)

    # フォールド番号を割り当て
    for n, (train_index, val_index) in enumerate(kf):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    
    print(train.groupby('fold').size())
    return train

def make_train_folds():
    """訓練用フォールドデータを作成・保存"""
    train_df = pd.read_csv(CFG.comp_dataset_path + 'train_features.csv')
    
    # シーン情報を抽出（IDの最初の部分）
    train_df['scene'] = train_df['ID'].str.split('_').str[0]
    
    print('group', CFG.group_col)
    print(f'train len: {len(train_df)}')
    
    # フォールド分割実行
    train_df = get_fold(train_df, CFG)
    print(train_df['fold'].value_counts())
    
    # フォールドデータ保存
    os.makedirs(CFG.train_fold_dir, exist_ok=True)
    train_df.to_csv(os.path.join(CFG.train_fold_dir, 'train_folds.csv'), index=False)
    print(f"Fold data saved to: {CFG.train_fold_dir}")

# ========== 基本設定関数 ==========
def set_seed(seed=None, cudnn_deterministic=True):
    """乱数シード固定"""
    if seed is None:
        seed = 42
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

def make_dirs(cfg):
    """出力ディレクトリ作成"""
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)

def cfg_init(cfg, mode='train'):
    """設定初期化"""
    set_seed(cfg.seed)
    if mode == 'train':
        make_dirs(cfg)

# ========== ログ設定 ==========
def init_logger(log_file):
    """ロガー初期化"""
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

class AverageMeter(object):
    """平均値を計算・保存するクラス"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    """秒を分秒形式に変換"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    """経過時間と残り時間を計算"""
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

# ========== 評価関数 ==========
def calc_mae_atmacup(y_true, y_pred):
    """atmaCup18用MAE計算"""
    abs_diff = np.abs(y_true - y_pred)  # 絶対誤差
    mae = np.mean(abs_diff.reshape(-1,))  # 平均絶対誤差
    return mae

def get_score(y_true, y_pred):
    """スコア計算"""
    eval_func = eval(CFG.metrics)
    return eval_func(y_true, y_pred)

def get_result(result_df):
    """結果計算・ログ出力"""
    pred_cols = [f'pred_{i}' for i in range(CFG.target_size)]
    preds = result_df[pred_cols].values
    labels = result_df[CFG.target_col].values
    score = get_score(labels, preds)
    Logger.info(f'score: {score:<.4f}')
    return score

# ========== 画像処理ユーティリティ ==========
def read_image_for_cache(path):
    """画像読み込み（キャッシュ用）"""
    if CFG.use_gray_scale:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (path, image)

def make_video_cache(paths):
    """画像キャッシュ作成（マルチプロセシング）"""
    debug = []
    for idx in range(9):
        color = 255 - int(255*(idx/9))
        debug.append(color)
    print(debug)

    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        res = pool.imap_unordered(read_image_for_cache, paths)
        res = tqdm(res)
        res = list(res)
    
    return dict(res)

# ========== データセット ==========
from albumentations import ReplayCompose
from torch.utils.data import DataLoader, Dataset

def get_transforms(data, cfg):
    """データ拡張取得"""
    if data == 'train':
        aug = A.ReplayCompose(cfg.train_aug_list)  # 訓練用拡張
    elif data == 'valid':
        aug = A.ReplayCompose(cfg.valid_aug_list)  # 検証用拡張
    return aug

class CustomDataset(Dataset):
    """カスタムデータセットクラス"""
    def __init__(self, df, cfg, labels=None, transform=None):
        self.df = df
        self.cfg = cfg
        self.base_paths = df['base_path'].values
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def read_image_multiframe(self, idx):
        """マルチフレーム画像読み込み"""
        base_path = self.base_paths[idx]
        
        images = []
        # 3つの時間フレーム（t-1.0, t-0.5, t）を読み込み
        suffixs = ['image_t-1.0.png', 'image_t-0.5.png', 'image_t.png']
        for suffix in suffixs:
            path = base_path + suffix
            image = self.cfg.video_cache[path]  # キャッシュから取得
            images.append(image)
        return images

    def __getitem__(self, idx):
        # マルチフレーム画像読み込み
        image = self.read_image_multiframe(idx)

        if self.transform:
            # 全フレームに同じ拡張を適用（ReplayCompose使用）
            replay = None
            images = []
            for img in image:
                if replay is None:
                    sample = self.transform(image=img)
                    replay = sample['replay']  # 拡張パラメータ保存
                else:
                    sample = ReplayCompose.replay(replay, image=img)  # 同じ拡張適用
                images.append(sample['image'])
            
            # 全フレームを結合（チャンネル方向）
            image = torch.concat(images, dim=0)

        if self.labels is None:
            return image

        # ラベル変換
        if self.cfg.objective_cv == 'multiclass':
            label = torch.tensor(self.labels[idx]).long()
        else:
            label = torch.tensor(self.labels[idx]).float()

        return image, label

# ========== モデル定義 ==========
import timm

class CustomModel(nn.Module):
    """ConvNeXtベースのカスタムモデル"""
    def __init__(self, cfg, pretrained=False, target_size=None, model_name=None):
        super().__init__()
        if model_name is None:
            model_name = "convnext_tiny"  # ConvNeXt Tinyを使用
        print(f'Using model: {model_name}, pretrained: {pretrained}')

        # ConvNeXtバックボーン初期化
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,  # 特徴抽出器として使用
            in_chans=cfg.model_in_chans  # 入力チャンネル数
        )
        
        # 特徴量次元数取得
        self.n_features = self.model.num_features

        # 回帰用ヘッド
        self.fc = nn.Sequential(
            nn.Linear(self.n_features, target_size or cfg.target_size)
        )

    def forward(self, image):
        # ConvNeXtで特徴抽出 → 線形層で回帰
        feature = self.model(image)
        output = self.fc(feature)
        return output

# ========== 学習率スケジューラ ==========
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """改良版Gradual Warmupスケジューラ"""
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    """スケジューラ取得"""
    if cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.factor, patience=cfg.patience, verbose=True, eps=cfg.eps)
    elif cfg.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr, last_epoch=-1)
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T_0, T_mult=1, eta_min=cfg.min_lr, last_epoch=-1)
    elif cfg.scheduler == 'GradualWarmupSchedulerV2':
        # コサインアニーリング + ウォームアップ
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.epochs, eta_min=1e-7)
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    
    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    """スケジューラステップ実行"""
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(avg_val_loss)
    elif isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()
    elif isinstance(scheduler, CosineAnnealingWarmRestarts):
        scheduler.step()
    elif isinstance(scheduler, GradualWarmupSchedulerV2):
        scheduler.step(epoch)

# ========== 訓練・検証関数 ==========
def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, model_ema=None):
    """1エポックの訓練"""
    model.train()
    scaler = GradScaler(enabled=CFG.use_amp)  # 混合精度用スケーラー
    
    losses = AverageMeter()
    preds = []
    preds_labels = []
    start = time.time()
    global_step = 0

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # 順伝播（混合精度）
        with autocast(CFG.use_amp):
            y_preds = model(images)
            if y_preds.size(1) == 1:
                y_preds = y_preds.view(-1)
            loss = criterion(y_preds, labels)

        # 損失更新
        losses.update(loss.item(), batch_size)
        
        # 逆伝播
        scaler.scale(loss).backward()
        
        # 勾配クリッピング
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        # オプティマイザ更新
        scaler.step(optimizer)
        scaler.update()
        
        # EMA更新
        if model_ema is not None:
            model_ema.update(model)
        
        optimizer.zero_grad()
        global_step += 1

        # 予測値保存
        if CFG.objective_cv == 'binary':
            preds.append(torch.sigmoid(y_preds).detach().to('cpu').numpy())
        elif CFG.objective_cv == 'multiclass':
            preds.append(y_preds.softmax(1).detach().to('cpu').numpy())
        elif CFG.objective_cv == 'regression':
            preds.append(y_preds.detach().to('cpu').numpy())
        
        preds_labels.append(labels.detach().to('cpu').numpy())

        # ログ出力
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
    
    predictions = np.concatenate(preds)
    labels = np.concatenate(preds_labels)
    return losses.avg, predictions, labels

def valid_fn(valid_loader, model, criterion, device):
    """検証関数"""
    model.eval()
    losses = AverageMeter()
    preds = []
    start = time.time()

    for step, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # 推論（勾配計算なし）
        with torch.no_grad():
            y_preds = model(images)

        if y_preds.size(1) == 1:
            y_preds = y_preds.view(-1)

        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # 予測値保存
        if CFG.objective_cv == 'binary':
            preds.append(torch.sigmoid(y_preds).to('cpu').numpy())
        elif CFG.objective_cv == 'multiclass':
            preds.append(y_preds.softmax(1).to('cpu').numpy())
        elif CFG.objective_cv == 'regression':
            preds.append(y_preds.to('cpu').numpy())

        # ログ出力
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step + 1) / len(valid_loader))))
    
    predictions = np.concatenate(preds)
    return losses.avg, predictions

def train_fold(folds, fold):
    """単一フォールドの訓練"""
    Logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # データローダー準備
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index  # 訓練インデックス
    val_idx = folds[folds['fold'] == fold].index   # 検証インデックス

    if CFG.use_alldata:
        train_folds = folds.copy().reset_index(drop=True)
    else:
        train_folds = folds.loc[trn_idx].reset_index(drop=True)

    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    # ラベル取得
    train_labels = train_folds[CFG.target_col].values
    valid_labels = valid_folds[CFG.target_col].values

    # データセット作成
    train_dataset = CustomDataset(
        train_folds, CFG, labels=train_labels, 
        transform=get_transforms(data='train', cfg=CFG))
    valid_dataset = CustomDataset(
        valid_folds, CFG, labels=valid_labels, 
        transform=get_transforms(data='valid', cfg=CFG))

    # データローダー作成
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, 
                              pin_memory=True, 
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers, 
                              pin_memory=True, 
                              drop_last=False)

    # ====================================================
    # モデル・オプティマイザ準備
    # ====================================================
    model = CustomModel(CFG, pretrained=CFG.pretrained)
    model.to(device)

    # EMA（Exponential Moving Average）
    if CFG.use_ema:
        model_ema = ModelEmaV2(model, decay=CFG.ema_decay)
    else:
        model_ema = None

    # オプティマイザ・スケジューラ
    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler = get_scheduler(CFG, optimizer)

    # ====================================================
    # 訓練ループ
    # ====================================================
    # 損失関数
    if CFG.objective_cv == 'binary':
        criterion = nn.BCEWithLogitsLoss()
    elif CFG.objective_cv == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    elif CFG.objective_cv == 'regression':
        criterion = nn.L1Loss()  # MAE損失

    # ベストスコア初期化
    if CFG.metric_direction == 'minimize':
        best_score = np.inf
    elif CFG.metric_direction == 'maximize':
        best_score = -1

    best_loss = np.inf

    # エポックループ
    for epoch in range(CFG.epochs):
        start_time = time.time()

        # 訓練
        avg_loss, train_preds, train_labels_epoch = train_fn(
            fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, model_ema)
        train_score = get_score(train_labels_epoch, train_preds)

        # 検証
        if model_ema is not None:
            avg_val_loss, valid_preds = valid_fn(valid_loader, model_ema.module, criterion, device)
        else:
            avg_val_loss, valid_preds = valid_fn(valid_loader, model, criterion, device)

        # スケジューラ更新
        scheduler_step(scheduler, avg_val_loss, epoch)

        # スコア計算
        score = get_score(valid_labels, valid_preds)
        elapsed = time.time() - start_time

        # ログ出力
        Logger.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        Logger.info(f'Epoch {epoch+1} - avg_train_Score: {train_score:.4f} avgScore: {score:.4f}')

        # ベストモデル更新判定
        if CFG.metric_direction == 'minimize':
            update_best = score < best_score
        elif CFG.metric_direction == 'maximize':
            update_best = score > best_score

        if update_best:
            best_loss = avg_val_loss
            best_score = score

            Logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            Logger.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')

            # ベストモデル保存
            if model_ema is not None:
                torch.save({'model': model_ema.module.state_dict(), 'preds': valid_preds},
                           CFG.model_dir + f'{CFG.model_name}_fold{fold}_best.pth')
            else:
                torch.save({'model': model.state_dict(), 'preds': valid_preds},
                           CFG.model_dir + f'{CFG.model_name}_fold{fold}_best.pth')

    # 最終モデル保存
    if model_ema is not None:
        torch.save({'model': model_ema.module.state_dict(), 'preds': valid_preds},
                   CFG.model_dir + f'{CFG.model_name}_fold{fold}_last.pth')
    else:
        torch.save({'model': model.state_dict(), 'preds': valid_preds},
                   CFG.model_dir + f'{CFG.model_name}_fold{fold}_last.pth')

    # ベストモデルの予測結果を返す
    check_point = torch.load(
        CFG.model_dir + f'{CFG.model_name}_fold{fold}_{CFG.inf_weight}.pth', 
        map_location=torch.device('cpu'))
    
    pred_cols = [f'pred_{i}' for i in range(CFG.target_size)]
    check_point_pred = check_point['preds']

    # 予測結果の形状調整
    if check_point_pred.ndim == 1:
        check_point_pred = check_point_pred.reshape(-1, CFG.target_size)

    print('check_point_pred shape', check_point_pred.shape)
    valid_folds[pred_cols] = check_point_pred
    
    return valid_folds

# ========== メイン訓練関数 ==========
def main():
    """メイン訓練処理"""
    # フォールドデータ読み込み
    train = pd.read_csv(CFG.train_fold_dir + 'train_folds.csv')
    train['ori_idx'] = train.index  # 元のインデックス保存
    train['scene'] = train['ID'].str.split('_').str[0]  # シーン情報抽出
    
    # 画像パス生成
    train['base_path'] = CFG.comp_dataset_path + 'images/' + train['ID'] + '/'

    # 全画像パスリスト作成（キャッシュ用）
    paths = []
    for base_path in train['base_path'].values:
        suffixs = ['image_t-1.0.png', 'image_t-0.5.png', 'image_t.png']
        for suffix in suffixs:
            path = base_path + suffix
            paths.append(path)

    print(paths[:5])

    # 画像キャッシュ作成（高速化のため）
    CFG.video_cache = make_video_cache(paths)

    # フォールド別訓練
    oof_df = pd.DataFrame()  # Out-of-Fold予測結果
    for fold in range(CFG.n_fold):
        if fold not in CFG.train_folds:
            print(f'fold {fold} is skipped')
            continue

        # 単一フォールド訓練
        _oof_df = train_fold(train, fold)
        oof_df = pd.concat([oof_df, _oof_df])
        
        # フォールド結果ログ
        Logger.info(f"========== fold: {fold} result ==========")
        get_result(_oof_df)

        # ホールドアウトまたは全データ使用時は1回のみ
        if CFG.use_holdout or CFG.use_alldata:
            break

    # OOF結果整理
    oof_df = oof_df.sort_values('ori_idx').reset_index(drop=True)

    # CV結果計算・ログ
    Logger.info("========== CV ==========")
    score = get_result(oof_df)

    # 結果保存
    oof_df.to_csv(CFG.submission_dir + 'oof_cv.csv', index=False)

# ========== アンサンブル推論クラス ==========
class EnsembleModel:
    """複数モデルのアンサンブル推論"""
    def __init__(self):
        self.models = []

    def __call__(self, x):
        outputs = []
        for model in self.models:
            if CFG.objective_cv == 'binary':
                outputs.append(torch.sigmoid(model(x)).to('cpu').numpy())
            elif CFG.objective_cv == 'multiclass':
                outputs.append(torch.softmax(model(x), axis=1).to('cpu').numpy())
            elif CFG.objective_cv == 'regression':
                outputs.append(model(x).to('cpu').numpy())

        # 平均アンサンブル
        avg_preds = np.mean(outputs, axis=0)
        return avg_preds

    def add_model(self, model):
        """モデル追加"""
        self.models.append(model)

def test_fn(valid_loader, model, device):
    """テストデータ推論"""
    preds = []

    for step, (images) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)

        preds.append(y_preds)

    predictions = np.concatenate(preds)
    return predictions

def inference():
    """テストデータ推論・提出ファイル作成"""
    # テストデータ読み込み
    test = pd.read_csv(CFG.comp_dataset_path + 'test_features.csv')
    test['base_path'] = CFG.comp_dataset_path + 'images/' + test['ID'] + '/'

    # テスト画像パス生成
    paths = []
    for base_path in test['base_path'].values:
        suffixs = ['image_t-1.0.png', 'image_t-0.5.png', 'image_t.png']
        for suffix in suffixs:
            path = base_path + suffix
            paths.append(path)

    print(paths[:5])

    # テスト画像キャッシュ作成
    CFG.video_cache = make_video_cache(paths)
    print(test.head(5))

    # テストデータセット・ローダー作成
    valid_dataset = CustomDataset(test, CFG, transform=get_transforms(data='valid', cfg=CFG))
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers, 
                              pin_memory=True, 
                              drop_last=False)

    # アンサンブルモデル構築
    model = EnsembleModel()
    folds = [0] if CFG.use_holdout else list(range(CFG.n_fold))
    
    for fold in folds:
        # 各フォールドのモデル読み込み
        _model = CustomModel(CFG, pretrained=False)
        _model.to(device)

        model_path = CFG.model_dir + f'{CFG.model_name}_fold{fold}_{CFG.inf_weight}.pth'
        print('load', model_path)
        state = torch.load(model_path)['model']
        _model.load_state_dict(state)
        _model.eval()

        model.add_model(_model)

    # 推論実行
    preds = test_fn(valid_loader, model, device)

    # 提出ファイル作成・保存
    test[CFG.target_col] = preds
    test.to_csv(CFG.submission_dir + 'submission_oof.csv', index=False)
    test[CFG.target_col].to_csv(CFG.submission_dir + f'submission_{CFG.exp_name}.csv', index=False)

# ========== 実行部分 ==========
if __name__ == "__main__":
    # 基本設定初期化
    cfg_init(CFG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ロガー初期化
    Logger = init_logger(log_file=CFG.log_path)
    Logger.info('\\-------- exp_info -----------------')
    Logger.info(datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'))
    
    # フォールド作成
    make_train_folds()
    
    # メイン訓練実行
    main()
    
    # 推論実行
    inference()
