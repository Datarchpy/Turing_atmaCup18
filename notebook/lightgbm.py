# AtmaCup18 LightGBM + CNN スタッキングベースライン
# 概要: カメラ情報をCNNで処理し、その予測値とテーブルデータをLightGBMでスタッキングする手法

import warnings
import pandas as pd
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import datetime
import math

# 設定クラス
class CFG:
    # ============== コンペ・実験設定 =============
    comp_name = 'atmacup_18'  # コンペ名
    comp_folder_name = 'atmacup_18'  # データセットのフォルダ名
    comp_dir_path = './'
    comp_dataset_path = '/kaggle/input/atmacup18-dataset/atmaCup18_dataset/'  # データセットのパス
    exp_name = 'atmacup_18_gbdt'  # 実験名

    # ============== ファイルパス =============
    train_fold_dir = "/kaggle/input/atmacup-18-cnn-exp001/atmacup_18_cnn_exp001/"  # CNNモデルの結果パス

    # ============== 予測ターゲット =============
    target_size = 18  # 予測対象の次元数（6つの物体 × xyz座標）
    target_col = ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2',
                  'z_2', 'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4', 'x_5', 'y_5', 'z_5']

    # ============== クロスバリデーション設定 =============
    n_fold = 3  # フォールド数
    skf_col = 'class'  # 層化分割用カラム
    group_col = 'scene'  # グループ分割用カラム
    fold_type = 'gkf'  # フォールドタイプ

    # ============== 評価指標設定 =============
    objective_cv = 'regression'  # 目的関数（回帰）
    metric_direction = 'minimize'  # 最適化方向
    metrics = 'calc_mae_atmacup'  # 評価関数

    # ============== 固定設定 =============
    seed = 42  # 乱数シード

    # ============== 出力パス設定 =============
    if exp_name is not None:
        print('set dataset path')
        outputs_path = comp_dir_path + f'outputs/{comp_name}/{exp_name}/'
        submission_dir = outputs_path + 'submissions/'
        submission_path = submission_dir + f'submission_{exp_name}.csv'
        model_dir = outputs_path + f'{comp_name}-models/'
        figures_dir = outputs_path + 'figures/'
        log_dir = outputs_path + 'logs/'
        log_path = log_dir + f'{exp_name}.txt'

# ============== 環境設定 =============
import torch, random

def set_seed(seed=None, cudnn_deterministic=True):
    """乱数シードを固定する関数"""
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
    """必要なディレクトリを作成する関数"""
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)

def cfg_init(cfg, mode='train'):
    """設定の初期化を行う関数"""
    set_seed(cfg.seed)
    if mode == 'train':
        make_dirs(cfg)

# 警告の無視と初期化
warnings.filterwarnings('ignore')
cfg_init(CFG)

# ============== ログ設定 =============
def init_logger(log_file):
    """ログの初期化を行う関数"""
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
    """値の平均を計算するクラス"""
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
    """秒を分:秒形式に変換する関数"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    """経過時間と残り時間を計算する関数"""
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

# ログの初期化
Logger = init_logger(log_file=CFG.log_path)
Logger.info('\\-------- exp_info -----------------')
Logger.info(datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'))

# ============== 評価関数 =============
def calc_mae_atmacup(y_true, y_pred):
    """AtmaCup18用のMAE計算関数"""
    abs_diff = np.abs(y_true - y_pred)  # 各予測の差分の絶対値を計算
    mae = np.mean(abs_diff.reshape(-1, ))  # 差分の絶対値の平均を計算
    return mae

def get_result(result_df):
    """結果を評価する関数"""
    pred_cols = [f'pred_{i}' for i in range(CFG.target_size)]
    preds = result_df[pred_cols].values
    labels = result_df[CFG.target_col].values
    
    eval_func = eval(CFG.metrics)
    best_score = eval_func(labels, preds)
    
    Logger.info(f'best_score: {best_score:<.4f}')
    return best_score

# ============== データ読み込み =============
# 訓練データの読み込み（フォールド情報付き）
train_df = pd.read_csv(CFG.train_fold_dir + 'train_folds.csv')

# データ数を制限（高速化のため最初の5000行のみ使用）
train_df = train_df.head(5000)

# テストデータの読み込み
test_df = pd.read_csv(CFG.comp_dataset_path + 'test_features.csv')

# ============== データ前処理 =============
def common_preprocess(target_df):
    """共通の前処理を行う関数"""
    # ブール型の列を整数型に変換
    bool_cols = ['brakePressed', 'gasPressed', 'leftBlinker', 'rightBlinker']
    print('bool_cols', bool_cols)
    target_df[bool_cols] = target_df[bool_cols].astype(int)

    # IDからシーン名と時刻を抽出
    target_df['scene'] = target_df['ID'].str.split('_').str[0]
    target_df['scene_sec'] = target_df['ID'].str.split('_').str[1].astype(int)

    # シーンごとのデータ数をカウント
    count_df = target_df.groupby('scene').size()
    target_df['scene_count'] = target_df['scene'].map(count_df)
    
    return target_df

# 前処理の適用
train_df = common_preprocess(train_df)
test_df = common_preprocess(test_df)

# ============== 信号機データの追加 =============
import os, json

# 信号機データの読み込み
ids = os.listdir(CFG.comp_dataset_path + 'traffic_lights')

traffic_lights = []
id_class_list = []

# 各ファイルから信号機情報を抽出
for id in ids:
    path = CFG.comp_dataset_path + f'traffic_lights/{id}'
    traffic_light = json.load(open(path))
    traffic_lights.append(traffic_light)
    
    for traffic_light in traffic_light:
        id_class_list.append((id.split('.')[0], traffic_light['class']))

# 信号機の状態データフレーム
traffic_lights_state_df = pd.DataFrame(id_class_list, columns=['ID', 'class'])

# 信号機の数データフレーム
counts = [len(traffic_light) for traffic_light in traffic_lights]
traffic_lights_count_df = pd.DataFrame({
    'ID': [id.split('.')[0] for id in ids],
    'traffic_lights_counts': counts
})

# 両方のデータフレームを結合
traffic_lights_df = pd.merge(traffic_lights_state_df, traffic_lights_count_df, on='ID', how='left')

# メインデータフレームに信号機データを結合
train_df = pd.merge(train_df, traffic_lights_df, on='ID', how='left')
test_df = pd.merge(test_df, traffic_lights_df, on='ID', how='left')

# 信号機クラスの分布確認
print("信号機クラスの分布:")
print(traffic_lights_df['class'].value_counts())

# ============== CNNの予測結果を追加 =============
exp_names = ['atmacup_18_cnn_exp001']
oof_feat_cols = []

for exp_name in exp_names:
    _oof_feat_cols = [f'{exp_name}_{c}' for c in CFG.target_col]   

    # CNN訓練データの予測結果を読み込み
    path = '/kaggle/input/atmacup-18-cnn-exp001/atmacup_18_cnn_exp001/submissions/oof_cv.csv'
    cnn_train_df = pd.read_csv(path)

    # 訓練データのIDに対応するCNN結果をフィルタリング
    cnn_train_df = cnn_train_df[cnn_train_df['ID'].isin(train_df['ID'])]
    
    # 訓練データのID順にCNN結果を並び替え
    cnn_train_df = cnn_train_df.set_index('ID').reindex(train_df['ID']).reset_index()
    cnn_train_df = cnn_train_df.reset_index(drop=True)
    
    # IDの対応確認
    print((train_df['ID'] == cnn_train_df['ID']).sum() / len(train_df))
    
    # CNN予測結果を訓練データに追加
    pred_cols = [f'pred_{i}' for i in range(CFG.target_size)]
    train_df[_oof_feat_cols] = cnn_train_df[pred_cols]
    print(_oof_feat_cols)

    # CNNテストデータの予測結果を読み込み
    path = '/kaggle/input/atmacup-18-cnn-exp001/atmacup_18_cnn_exp001/submissions/submission_oof.csv'
    cnn_test_df = pd.read_csv(path)
    pred_cols = CFG.target_col

    # CNN予測結果をテストデータに追加
    test_df[_oof_feat_cols] = cnn_test_df[pred_cols]
    oof_feat_cols.extend(_oof_feat_cols)

# ============== シフト特徴量作成 =============
def make_shift_feature(target_df, use_feat_cols):
    """時系列のシフト特徴量を作成する関数"""
    shift_count = 1
    shift_range = list(range(-shift_count, shift_count+1))
    shift_range = [x for x in shift_range if x != 0]

    target_df['ori_idx'] = target_df.index
    
    # シーンと時刻でソート
    target_df = target_df.sort_values(['scene', 'scene_sec']).reset_index(drop=True)

    shift_feat_cols = []
    for shift in shift_range:
        for col in use_feat_cols:
            # シフト特徴量
            shift_col = f'{col}_shift{shift}'
            target_df[shift_col] = target_df.groupby('scene')[col].shift(shift)
            shift_feat_cols.append(shift_col)

            # 差分特徴量
            diff_col = f'{col}_diff{shift}'
            target_df[diff_col] = target_df[col] - target_df[shift_col]
            shift_feat_cols.append(diff_col)

    # 元の順序に戻す
    target_df = target_df.sort_values('ori_idx').reset_index(drop=True)
    target_df = target_df.drop('ori_idx', axis=1)

    return target_df, shift_feat_cols

# ============== 追加特徴量作成関数 =============
def add_lag_features(df, cols, group_col='scene'):
    """ラグ特徴量と差分を作成する関数"""
    for col in cols:
        df[f'lag_{col}'] = df.groupby(group_col)[col].shift(1)
        df[f'diff_{col}'] = df[col] - df[f'lag_{col}']
    return df

def add_scene_agg_features(df, cols, group_col='scene'):
    """シーン単位の集約特徴量を追加する関数"""
    group = df.groupby(group_col)
    for col in cols:
        df[f'{col}_mean'] = group[col].transform('mean')
        df[f'{col}_std'] = group[col].transform('std')
        df[f'{col}_max'] = group[col].transform('max')
        df[f'{col}_min'] = group[col].transform('min')
    return df

def add_signal_interaction_features(df):
    """信号機との相互作用特徴量を作成する関数"""
    # 信号の状態（ワンホットエンコード）
    df['signal_red'] = (df['class'] == 'red').astype(int)
    df['signal_green'] = (df['class'] == 'green').astype(int)

    # 信号機のカウント
    df['traffic_lights_counts'] = df['traffic_lights_counts'].fillna(0).astype(int)

    # 信号の状態と速度の相互作用特徴量
    df['speed_signal_red'] = df['vEgo'] * df['signal_red']
    df['speed_signal_green'] = df['vEgo'] * df['signal_green']

    return df

def add_speed_steering_interaction(df):
    """速度と舵角の相互作用特徴量を作成する関数"""
    df['speed_steering_interaction'] = df['vEgo'] * df['steeringAngleDeg']
    return df

def apply_features(df, signal_df=None):
    """全ての特徴量を統合して適用する関数"""
    # ラグ特徴量・変化率
    lag_cols = ['vEgo', 'aEgo']
    df = add_lag_features(df, lag_cols)

    # シーン単位の集約特徴量
    scene_cols = ['vEgo', 'aEgo']
    df = add_scene_agg_features(df, scene_cols)

    # 信号機との相互作用特徴量
    df = add_signal_interaction_features(df)

    # 速度×舵角の相互作用特徴量
    df = add_speed_steering_interaction(df)

    return df

# ============== 特徴量ブロック定義 =============
from sklearn.preprocessing import LabelEncoder
from typing import List

class AbstractBaseBlock:
    """特徴量変換の基底クラス"""
    def __init__(self) -> None:
        pass

    def fit(self, input_df: pd.DataFrame, y=None) -> pd.DataFrame:
        raise NotImplementedError()

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

def run_block(input_df: pd.DataFrame, blocks: List[AbstractBaseBlock], is_fit):
    """特徴量ブロックを実行する関数"""
    output_df = pd.DataFrame()
    for block in blocks:
        name = block.__class__.__name__
        
        if is_fit:
            _df = block.fit(input_df)
        else:
            _df = block.transform(input_df)
        
        output_df = pd.concat([output_df, _df], axis=1)
    return output_df

class NumericBlock(AbstractBaseBlock):
    """数値特徴量ブロック"""
    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.col] = input_df[self.col].copy()
        return output_df

class LabelEncodingBlock(AbstractBaseBlock):
    """ラベルエンコーディングブロック"""
    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col
        self.encoder = LabelEncoder()

    def fit(self, input_df):
        self.encoder.fit(input_df[self.col])
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.col] = self.encoder.transform(input_df[self.col])
        return output_df.add_suffix('@le')

class CountEncodingBlock(AbstractBaseBlock):
    """カウントエンコーディングブロック"""
    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col

    def fit(self, input_df):
        self.val_count_dict = {}
        self.val_count = input_df[self.col].value_counts()
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.col] = input_df[self.col].map(self.val_count)
        return output_df.add_suffix('@ce')

# ============== 特徴量作成実行 =============
# 使用する基本特徴量
use_cols = ['vEgo', 'aEgo', 'steeringAngleDeg', 'steeringTorque', 'brake',
           'brakePressed', 'gas', 'gasPressed', 'leftBlinker', 'rightBlinker']
use_cols += oof_feat_cols

# シフト特徴量を作成
train_df, shift_feat_cols = make_shift_feature(train_df, use_cols)
test_df, shift_feat_cols = make_shift_feature(test_df, use_cols)

# 追加特徴量を適用
train_df = apply_features(train_df, signal_df=traffic_lights_df)
test_df = apply_features(test_df, signal_df=traffic_lights_df)

# ============== 最終的な特徴量設定 =============
# 基本の数値特徴量
base_cols = ['vEgo', 'aEgo', 'steeringAngleDeg', 'steeringTorque', 'brake',
             'brakePressed', 'gas', 'gasPressed', 'leftBlinker', 'rightBlinker', 'scene_sec']

# 新規の特徴量
lag_cols = [f'lag_{col}' for col in ['vEgo', 'aEgo']]
diff_cols = [f'diff_{col}' for col in ['vEgo', 'aEgo']]
agg_cols = [f'{col}_{stat}' for col in ['vEgo', 'aEgo'] for stat in ['mean', 'std', 'max', 'min']]
interaction_cols = ['speed_steering_interaction']
signal_interaction_cols = ['signal_red', 'signal_green', 'speed_signal_red', 
                          'speed_signal_green', 'traffic_lights_counts']

# 特徴量フラグ（実験時に有効/無効を切り替え可能）
USE_LAG_FEATURES = True
USE_DIFF_FEATURES = True
USE_AGG_FEATURES = True
USE_INTERACTION_FEATURES = True
USE_SIGNAL_FEATURES = True

# 使用する特徴量を動的に構築
num_cols = base_cols + oof_feat_cols + shift_feat_cols + ['scene_count']
if USE_LAG_FEATURES:
    num_cols += lag_cols
if USE_DIFF_FEATURES:
    num_cols += diff_cols
if USE_AGG_FEATURES:
    num_cols += agg_cols
if USE_INTERACTION_FEATURES:
    num_cols += interaction_cols
if USE_SIGNAL_FEATURES:
    num_cols += signal_interaction_cols

print(f'使用する数値特徴量: {len(num_cols)}個')

# 訓練・テストデータを結合して特徴量エンコーディング
train_num = len(train_df)
whole_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# カテゴリ変数の定義
cat_label_cols = ['gearShifter']
cat_count_cols = []

# 特徴量ブロックの設定
blocks = [
    *[NumericBlock(col) for col in num_cols],
    *[LabelEncodingBlock(col) for col in cat_label_cols],
    *[CountEncodingBlock(col) for col in cat_count_cols],
]

# 特徴量の構築
whole_feat_df = run_block(whole_df, blocks, is_fit=True)

# 訓練・テストデータに分割
train_df, test_df = whole_df.iloc[:train_num], whole_df.iloc[train_num:].drop(
    columns=CFG.target_col).reset_index(drop=True)
train_feat, test_feat = whole_feat_df.iloc[:train_num], whole_feat_df.iloc[train_num:].reset_index(drop=True)

print('最終的に使用する特徴量数:', len(train_feat.columns))

# ターゲットとフォールド情報の設定
y = train_df[CFG.target_col]
folds = train_df['fold']

# 欠損値の確認
print("訓練データの欠損値:")
print(train_feat.isnull().sum().sum())
print("テストデータの欠損値:")
print(test_feat.isnull().sum().sum())

# ============== LightGBMモデル定義 =============
import lightgbm as lgb

class LightGBM:
    """LightGBMのラッパークラス"""
    
    def __init__(self, lgb_params, save_dir=None, imp_dir=None, categorical_feature=None,
                 model_name='lgb', stopping_rounds=50) -> None:
        self.save_dir = save_dir
        self.imp_dir = imp_dir
        self.lgb_params = lgb_params
        self.categorical_feature = categorical_feature
        self.model_name = model_name
        self.stopping_rounds = stopping_rounds

    def fit(self, x_train, y_train, **fit_params) -> None:
        """モデルを訓練する"""
        X_val, y_val = fit_params['eval_set'][0]
        del fit_params['eval_set']

        train_dataset = lgb.Dataset(x_train, y_train, categorical_feature=self.categorical_feature)
        val_dataset = lgb.Dataset(X_val, y_val, categorical_feature=self.categorical_feature)

        self.model = lgb.train(
            params=self.lgb_params,
            train_set=train_dataset,
            valid_sets=[train_dataset, val_dataset],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.stopping_rounds, verbose=True),
                lgb.log_evaluation(500)
            ],
            **fit_params
        )

    def save(self, fold):
        """モデルを保存する"""
        save_to = f'{self.save_dir}lgb_fold_{fold}_{self.model_name}.txt'
        self.model.save_model(save_to)

    def predict(self, x):
        """予測を行う"""
        return self.model.predict(x)

def get_model(model_name):
    """LightGBMモデルを取得する関数"""
    lgb_params = {
        'objective': CFG.objective_cv,
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': 8,
        'seed': CFG.seed,
        'learning_rate': 0.01,
        'metric': 'mae',
        'num_leaves': 64,  # 葉の数
        'max_depth': 5,    # 木の深さ
        'bagging_seed': CFG.seed,
        'feature_fraction_seed': CFG.seed,
        'drop_seed': CFG.seed,
    }
    
    model = LightGBM(
        lgb_params=lgb_params,
        imp_dir=CFG.figures_dir,
        save_dir=CFG.model_dir,
        model_name=model_name
    )
    
    return model

def get_fit_params(model_name):
    """モデル訓練パラメータを取得する関数"""
    params = {'num_boost_round': 100000}
    return params

# ============== メイン訓練・予測処理 =============
def main(train_df, X_train, y, folds, test_df):
    """メインの訓練・予測処理を行う関数"""
    eval_func = eval(CFG.metrics)

    # 予測結果を格納する配列
    oof_predictions = np.zeros((X_train.shape[0], CFG.target_size))
    test_predictions = np.zeros((test_df.shape[0], CFG.target_size))

    # 各ターゲット（18次元）について個別にモデルを訓練
    for target_idx in range(CFG.target_size):
        Logger.info(f'target {target_idx}')

        # 各フォールドでクロスバリデーション
        for fold in range(CFG.n_fold):
            Logger.info(f'Training fold {fold + 1}')
            target_col = CFG.target_col[target_idx]

            # モデルの初期化
            model_name = f'lgb_{target_col}'
            model = get_model(model_name)
            fit_params = get_fit_params(model_name)

            # 訓練・検証データの分割
            trn_ind = folds != fold
            val_ind = folds == fold

            x_train, x_val = X_train.loc[trn_ind], X_train.loc[val_ind]
            y_train, y_val = y.loc[trn_ind, target_col], y.loc[val_ind, target_col]
            eval_set = [(x_val, y_val)]

            fit_params_fold = fit_params.copy()
            fit_params_fold['eval_set'] = eval_set

            # モデル訓練
            model.fit(x_train, y_train, **fit_params_fold)

            # モデル保存
            if hasattr(model, 'save'):
                model.save(fold)

            # Out-of-Fold予測
            oof_predictions[val_ind, target_idx] = model.predict(x_val)

            # テストデータ予測（後でアンサンブル平均を取る）
            test_predictions[:, target_idx] += model.predict(test_df)

    # Out-of-Fold評価
    score = eval_func(y.values, oof_predictions)
    Logger.info(f'oof result {score}')

    # 結果の保存
    pred_cols = [f'pred_{i}' for i in range(CFG.target_size)]

    oof = train_df.copy()
    oof[pred_cols] = oof_predictions
    oof[CFG.target_col] = y

    oof_feat = X_train.copy()
    oof_feat[pred_cols] = oof_predictions
    oof_feat[CFG.target_col] = y

    # 結果評価
    get_result(oof)

    # CSV保存
    oof.to_csv(CFG.submission_dir + 'oof_gbdt.csv', index=False)
    oof_feat.to_csv(CFG.submission_dir + 'oof_feat_gbdt.csv', index=False)

    # テスト予測の平均化（フォールド数で割る）
    test_predictions /= CFG.n_fold

    # テスト結果の保存
    test_df[CFG.target_col] = test_predictions
    test_df.to_csv(CFG.submission_dir + 'submission_oof.csv', index=False)
    test_df[CFG.target_col].to_csv(CFG.submission_dir + f'submission_{CFG.exp_name}.csv', index=False)

    # サンプル提出ファイルとの比較
    sample_sub = pd.read_csv('/kaggle/input/atmacup18-sample-submit/atmaCup18__sample_submit.csv')
    print('sample_sub_len: ', len(sample_sub))
    print('sub_len: ', len(test_df))

# ============== 実行 =============
if __name__ == "__main__":
    # メイン処理の実行
    main(train_df, train_feat, y, folds, test_feat)

    # ============== 結果の詳細分析 =============
    # Out-of-Fold結果の読み込み
    oof = pd.read_csv(CFG.submission_dir + 'oof_feat_gbdt.csv')
    sub_oof = pd.read_csv(CFG.submission_dir + f'submission_oof.csv')

    # 各ターゲットごとのスコア詳細
    print("\n=== 各ターゲットごとのMAEスコア ===")
    for i, col in enumerate(CFG.target_col):
        y_true = oof[col].values
        y_pred = oof[f'pred_{i}'].values
        
        score = calc_mae_atmacup(y_true, y_pred)
        Logger.info(f'{col} score: {score}')
        
        # 物体別・座標別の分析
        object_num = i // 3  # 物体番号（0-5）
        coord = ['x', 'y', 'z'][i % 3]  # 座標軸
        print(f"物体{object_num}の{coord}座標: MAE = {score:.4f}")

    print(f"\n=== 物体別の平均MAE ===")
    for obj_idx in range(6):  # 6つの物体
        obj_scores = []
        for coord_idx in range(3):  # x, y, z座標
            target_idx = obj_idx * 3 + coord_idx
            col = CFG.target_col[target_idx]
            y_true = oof[col].values
            y_pred = oof[f'pred_{target_idx}'].values
            score = calc_mae_atmacup(y_true, y_pred)
            obj_scores.append(score)
        
        avg_score = np.mean(obj_scores)
        print(f"物体{obj_idx}: 平均MAE = {avg_score:.4f} (x:{obj_scores[0]:.3f}, y:{obj_scores[1]:.3f}, z:{obj_scores[2]:.3f})")

    print(f"\n=== 座標軸別の平均MAE ===")
    for coord_idx, coord_name in enumerate(['x', 'y', 'z']):
        coord_scores = []
        for obj_idx in range(6):
            target_idx = obj_idx * 3 + coord_idx
            col = CFG.target_col[target_idx]
            y_true = oof[col].values
            y_pred = oof[f'pred_{target_idx}'].values
            score = calc_mae_atmacup(y_true, y_pred)
            coord_scores.append(score)
        
        avg_score = np.mean(coord_scores)
        print(f"{coord_name}座標: 平均MAE = {avg_score:.4f}")

    # ============== 特徴量重要度の分析（参考） =============
    print(f"\n=== 実験設定のまとめ ===")
    print(f"使用データ数: {len(train_df):,}行")
    print(f"特徴量数: {len(train_feat.columns)}個")
    print(f"フォールド数: {CFG.n_fold}")
    print(f"予測ターゲット数: {CFG.target_size}")
    print(f"評価指標: {CFG.metrics}")
    print(f"実験名: {CFG.exp_name}")
    
    print(f"\n=== 使用した特徴量カテゴリ ===")
    print(f"基本特徴量: {len(base_cols)}個")
    print(f"CNN予測特徴量: {len(oof_feat_cols)}個")
    print(f"シフト特徴量: {len(shift_feat_cols)}個")
    if USE_LAG_FEATURES:
        print(f"ラグ特徴量: {len(lag_cols)}個")
    if USE_DIFF_FEATURES:
        print(f"差分特徴量: {len(diff_cols)}個")
    if USE_AGG_FEATURES:
        print(f"集約特徴量: {len(agg_cols)}個")
    if USE_INTERACTION_FEATURES:
        print(f"相互作用特徴量: {len(interaction_cols)}個")
    if USE_SIGNAL_FEATURES:
        print(f"信号機特徴量: {len(signal_interaction_cols)}個")

    print(f"\n=== 今後の改善案 ===")
    print("1. ハイパーパラメータチューニング（Optuna等を使用）")
    print("2. より複雑な時系列特徴量（移動平均、指数平滑化等）")
    print("3. 物体間の距離・角度関係の特徴量")
    print("4. CNNモデルの改善（異なるアーキテクチャのアンサンブル）")
    print("5. 深度画像やその他のセンサーデータの活用")
    print("6. より高度な特徴量選択手法")
    print("7. ニューラルネットワークとの組み合わせ（Neural Network + GBDT）")

    Logger.info('\\-------- 実験完了 -----------------')
