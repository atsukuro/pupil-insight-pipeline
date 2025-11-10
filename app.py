# app.py (v26.0.2 - KeyError修正版)
# 変更点:
# 1. 【バグ修正】動画評価のExcelファイル作成時に発生していた 'timestamp' の
#    KeyErrorを修正。フレーム番号からタイムスタンプを再計算して追加する
#    処理を追加。
# 2. 【動画評価 機能拡張】出力ファイル形式をExcelに変更し、
#    エンゲージメントサマリーシートを追加する機能は維持。

import os
import gc
import traceback
import threading
import pandas as pd
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from flask import Flask, request, jsonify
from moviepy.editor import VideoFileClip
from gaze_mapper import GazeMapper
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import time
import json
import glob

# =============================================================================
# SECTION 1: 定数とグローバル設定
# =============================================================================
app = Flask(__name__)
BASE_PATH = os.path.dirname(os.path.abspath(__file__)); STIMULI_PATH = os.path.join(BASE_PATH, "stimuli"); VIDEOS_PATH = os.path.join(BASE_PATH, "videos"); OUTPUTS_PATH = os.path.join(BASE_PATH, "analysis_outputs"); FONT_PATH = os.path.join(BASE_PATH, 'NotoSansJP-Regular.ttf')
os.makedirs(STIMULI_PATH, exist_ok=True); os.makedirs(VIDEOS_PATH, exist_ok=True); os.makedirs(OUTPUTS_PATH, exist_ok=True)
jp_font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None

# =============================================================================
# SECTION 1.5: グラフ・画像生成ヘルパー関数
# =============================================================================
def get_foveal_luminance(frame, gaze_x, gaze_y, fovea_radius=50):
    if frame is None or pd.isna(gaze_x) or pd.isna(gaze_y): return np.nan
    h, w = frame.shape[:2]; x, y = int(gaze_x), int(gaze_y)
    x_min, x_max = max(0, x - fovea_radius), min(w, x + fovea_radius)
    y_min, y_max = max(0, y - fovea_radius), min(h, y + fovea_radius)
    if x_min >= x_max or y_min >= y_max: return np.nan
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0: return np.nan
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) > 2 and roi.shape[2] == 3 else roi
    return np.mean(gray_roi)

def generate_pre_work_dashboard(gaze_df, results, output_path, historical_data):
    fig, axs = plt.subplots(2, 2, figsize=(20, 15), facecolor='whitesmoke')
    fig.suptitle('就業前チェック結果ダッシュボード', fontsize=28, fontproperties=jp_font)
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    ax1.set_title('眼球運動速度の推移', fontproperties=jp_font, fontsize=16)
    if not historical_data.empty:
        historical_data['datetime'] = pd.to_datetime(historical_data['timestamp'], unit='s')
        ax1.plot(historical_data['datetime'], historical_data['saccade_mean_speed'], marker='o', linestyle='-', label='平均速度')
        ax1.plot(historical_data['datetime'], historical_data['saccade_max_speed'], marker='s', linestyle='--', label='最大速度')
    ax1.set_ylabel('速度 (pixels/sec)', fontproperties=jp_font)
    ax1.legend(prop=jp_font)

    ax2.set_title('対光反射の推移', fontproperties=jp_font, fontsize=16)
    if not historical_data.empty:
        ax2.plot(historical_data['datetime'], historical_data['plr_latency'], marker='o', linestyle='-', color='tab:green', label='潜時 (秒)')
        ax2.set_ylabel('潜時 (秒、値が低いほど良い)', fontproperties=jp_font, color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        
        ax2b = ax2.twinx()
        ax2b.plot(historical_data['datetime'], historical_data['plr_response'], marker='s', linestyle='--', color='tab:orange', label='応答率 (%)')
        ax2b.set_ylabel('応答率 (値が高いほど良い)', fontproperties=jp_font, color='tab:orange')
        ax2b.tick_params(axis='y', labelcolor='tab:orange')
        
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2b.legend(lines + lines2, labels + labels2, loc='upper right', prop=jp_font)

    ax3.set_title('固視安定性の推移', fontproperties=jp_font, fontsize=16)
    if not historical_data.empty:
        ax3.plot(historical_data['datetime'], historical_data['fixation_stability_score'], marker='o', linestyle='-', color='tab:purple', label='安定性スコア')
    ax3.set_ylabel('安定性スコア (値が低いほど良い)', fontproperties=jp_font)
    ax3.legend(prop=jp_font)
    
    ax4.set_title('今回の測定結果サマリー', fontproperties=jp_font, fontsize=16)
    ax4.axis('off')
    summary_text = (f"サッケード平均速度: {results.get('saccade_mean_speed', 0):.1f} px/s\n"
                    f"サッケード最大速度: {results.get('saccade_max_speed', 0):.1f} px/s\n\n"
                    f"対光反射 応答率: {results.get('plr_response', 0):.2%}\n"
                    f"対光反射 潜時: {results.get('plr_latency', 0):.3f} 秒\n\n"
                    f"固視安定性スコア: {results.get('fixation_stability_score', 0):.3f}")
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=15, fontproperties=jp_font, linespacing=1.8, bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="b", lw=1))

    for ax in [ax1, ax2, ax3]:
        if not historical_data.empty:
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            ax.text(0.5, 0.5, '過去のデータがありません\n(2回目以降の測定で表示)', ha='center', va='center', fontproperties=jp_font, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"--- 📊 就業前チェックのダッシュボードを生成しました: {output_path} ---")


# =============================================================================
# SECTION 2: 瞳孔ベースラインモデル
# =============================================================================
class PupilBaselineModel:
    def __init__(self, degree=2):
        self.poly = PolynomialFeatures(degree=degree); self.model_light = LinearRegression(); self.model_dark = LinearRegression(); self.is_trained = False
    def train(self, lum_calib_data):
        df = pd.DataFrame(lum_calib_data).dropna()
        if len(df) < 6: return False
        df_light, df_dark = df[df['direction'] == 'light'], df[df['direction'] == 'dark']
        if len(df_light) < 3 or len(df_dark) < 3: return False
        self.model_light.fit(self.poly.fit_transform(df_light[['luminance']]), df_light['pupil_radius'])
        self.model_dark.fit(self.poly.fit_transform(df_dark[['luminance']]), df_dark['pupil_radius'])
        self.is_trained = True; return True
    def predict(self, luminance, trend):
        if not self.is_trained: return np.nan
        lum_poly = self.poly.transform([[luminance]])
        return self.model_light.predict(lum_poly)[0] if trend == 'increasing' else self.model_dark.predict(lum_poly)[0]

# =============================================================================
# SECTION 3: 分析ロジック本体
# =============================================================================
def correct_gaze_data(gaze_df, calib_df, screen_w, screen_h):
    mapper = GazeMapper(screen_w, screen_h)
    if not mapper.train(calib_df):
        print("--- 警告: モデル訓練失敗。補正なし。 ---")
        gaze_df[['pred_x', 'pred_y']] = pd.DataFrame(gaze_df['raw_gaze_vector'].tolist(), index=gaze_df.index)
        return gaze_df
    pred_df = mapper.predict_bulk(gaze_df)
    gaze_df[['pred_x', 'pred_y']] = pred_df[['pred_x', 'pred_y']]
    
    margin = 90
    gaze_df['pred_x'] = gaze_df['pred_x'].clip(-margin, screen_w + margin)
    gaze_df['pred_y'] = gaze_df['pred_y'].clip(-margin, screen_h + margin)
    
    print(f"--- ✅ 高精度な視線座標の補正完了。 ---"); return gaze_df

def perform_heatmap_analysis(gaze_df, calib_df, lum_calib_data, item_name, item_path, screen_w, screen_h):
    print(f"--- 🔥 静止画 詳細分析開始: {item_name} ---")
    if not os.path.exists(item_path): print(f"--- エラー: 静止画なし: {item_path} ---"); return
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    corrected_df = correct_gaze_data(gaze_df.copy(), calib_df, screen_w, screen_h)
    
    baseline_df = corrected_df[corrected_df['timestamp'] < 3.0]
    if not baseline_df.empty and baseline_df['pupil_radius'].std() > 0:
        baseline_mean = baseline_df['pupil_radius'].mean(); baseline_std = baseline_df['pupil_radius'].std()
        corrected_df['pzs'] = (corrected_df['pupil_radius'] - baseline_mean) / (baseline_std if baseline_std > 0 else 1.0)
    else: corrected_df['pzs'] = 0

    image = cv2.imdecode(np.fromfile(item_path, np.uint8), cv2.IMREAD_COLOR)
    
    gaze_points = list(zip(corrected_df['pred_x'].astype(int), corrected_df['pred_y'].astype(int)))
    heatmap_img = generate_static_heatmap(image.copy(), gaze_points, screen_w, screen_h)
    output_heatmap = os.path.join(OUTPUTS_PATH, f"result_{os.path.splitext(item_name)[0]}_heatmap_{timestamp}.png")
    cv2.imwrite(output_heatmap, heatmap_img)
    print(f"--- ヒートマップ(滞在時間)画像生成完了: {output_heatmap} ---")

    pzs_heatmap_img = generate_pzs_heatmap(image.copy(), corrected_df, screen_w, screen_h)
    output_pzs_heatmap = os.path.join(OUTPUTS_PATH, f"result_{os.path.splitext(item_name)[0]}_pzs_heatmap_{timestamp}.png")
    cv2.imwrite(output_pzs_heatmap, pzs_heatmap_img)
    print(f"--- ヒートマップ(Zスコア)画像生成完了: {output_pzs_heatmap} ---")

    output_cognitiveload = os.path.join(OUTPUTS_PATH, f"result_{os.path.splitext(item_name)[0]}_cognitiveload_{timestamp}.png")
    load_fig = generate_cognitive_load_graph(corrected_df, os.path.splitext(item_name)[0])
    load_fig.savefig(output_cognitiveload, dpi=150); plt.close(load_fig)
    print(f"--- グラフ(Cognitive Load)生成完了: {output_cognitiveload} ---")

def generate_static_heatmap(image, gaze_points, screen_w, screen_h):
    h, w, _ = image.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    gaze_points_img = [(int(p[0] * w / screen_w), int(p[1] * h / screen_h)) for p in gaze_points]
    for x, y in gaze_points_img:
        if 0 <= x < w and 0 <= y < h:
            heatmap[y, x] += 1
    heatmap = cv2.GaussianBlur(heatmap, (151, 151), 30)
    if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

def generate_pzs_heatmap(image, df, screen_w, screen_h):
    h, w, _ = image.shape
    pzs_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.int32)
    for _, row in df.iterrows():
        x = int(row['pred_x'] * w / screen_w)
        y = int(row['pred_y'] * h / screen_h)
        if 0 <= x < w and 0 <= y < h and pd.notna(row['pzs']):
            pzs_map[y, x] += row['pzs']
            count_map[y, x] += 1
    pzs_map = np.divide(pzs_map, count_map, out=np.zeros_like(pzs_map), where=count_map!=0)
    pzs_map = cv2.GaussianBlur(pzs_map, (151, 151), 30)
    vmax = np.max(np.abs(pzs_map))
    if vmax > 0:
        pzs_map_norm = (pzs_map / (2 * vmax) + 0.5) * 255
    else:
        pzs_map_norm = np.full((h,w), 128, dtype=np.uint8)
    heatmap_colored = cv2.applyColorMap(pzs_map_norm.astype(np.uint8), cv2.COLORMAP_WINTER)
    return cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)

def generate_cognitive_load_graph(slide_df, slide_name):
    df = slide_df.copy()
    df['elapsed_time'] = df['timestamp'] - df['timestamp'].min()
    df['gaze_velocity'] = np.sqrt(df['pred_x'].diff()**2 + df['pred_y'].diff()**2).fillna(0)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color1 = 'tab:blue'
    ax1.set_xlabel('時間 (秒)', fontproperties=jp_font); ax1.set_ylabel('瞳孔反応 (PZS)', color=color1, fontproperties=jp_font)
    ax1.plot(df['elapsed_time'], df['pzs'], color=color1, label='瞳孔反応 (PZS)')
    ax1.tick_params(axis='y', labelcolor=color1); ax1.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax2 = ax1.twinx(); color2 = 'tab:red'
    ax2.set_ylabel('視線速度 (pixels/frame)', color=color2, fontproperties=jp_font)
    ax2.plot(df['elapsed_time'], df['gaze_velocity'], color=color2, label='視線速度', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color2)
    fig.suptitle(f'{slide_name} の認知的負荷', fontsize=16, fontproperties=jp_font)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def create_analyzed_video(video_path, df, output_path):
    video_cap = cv2.VideoCapture(video_path); width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = video_cap.get(cv2.CAP_PROP_FPS); total_frames = len(df)
    def get_pzs_color_bgr(pzs):
        if pzs > 2.0 or pd.isna(pzs): return None
        if pzs <= 0: return (255, 255, 255)
        if pzs > 1.2: return (0, 0, 255)
        intensity = int(100 + (pzs / 1.2) * 155); return (0, 0, intensity)
    fig, ax = plt.subplots(figsize=(width / 100, 2.5), dpi=100); graph_h = int(fig.get_figheight() * fig.dpi); new_height = height + graph_h
    df['pzs'] = pd.to_numeric(df['pzs'], errors='coerce')
    pzs_max, pzs_min = max(df['pzs'].quantile(0.98), 1.5), min(df['pzs'].quantile(0.02), -1.5)
    for index, row in df.iterrows():
        color_bgr = get_pzs_color_bgr(row['pzs'])
        if color_bgr: ax.bar(index, row['pzs'], color=tuple(c/255 for c in color_bgr[::-1]), width=1.0)
    ax.axhline(y=0.0, color='white', linestyle='--', linewidth=1); ax.set_xlim(0, len(df)); ax.set_ylim(pzs_min, pzs_max); ax.set_title('瞳孔反応 (PZS)', color='white', fontsize=16, fontproperties=jp_font)
    ax.tick_params(colors='white'); ax.grid(True, linestyle='--', alpha=0.2); fig.patch.set_facecolor('black'); ax.patch.set_facecolor('#181818'); fig.tight_layout()
    fig.canvas.draw(); graph_bg_bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR); graph_bg_resized = cv2.resize(graph_bg_bgr, (width, graph_h)); plt.close(fig)
    temp_video_path = output_path.replace('.mp4', '_temp_video.mp4'); fourcc = cv2.VideoWriter_fourcc(*'mp4v'); writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, new_height))
    for frame_index in tqdm(range(total_frames), desc="可視化ビデオを生成中"):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index); ret, frame_bgr = video_cap.read()
        if not ret: break
        canvas = np.zeros((new_height, width, 3), dtype=np.uint8); row = df.iloc[frame_index]; gaze_x, gaze_y, pzs = row['gaze_x_smooth'], row['gaze_y_smooth'], row['pzs']
        color = get_pzs_color_bgr(pzs)
        if color and pd.notna(gaze_x) and gaze_x > 0: cv2.circle(frame_bgr, (int(gaze_x), int(gaze_y)), 40, color, 3)
        canvas[0:height, 0:width] = frame_bgr; graph_img = graph_bg_resized.copy(); marker_x = int((frame_index / total_frames) * width) if total_frames > 0 else 0
        cv2.line(graph_img, (marker_x, 0), (marker_x, graph_h), (0, 255, 255), 2); canvas[height:new_height, :] = graph_img
        writer.write(canvas)
    writer.release(); video_cap.release()
    try:
        with VideoFileClip(video_path) as original_clip:
            if original_clip.audio:
                with VideoFileClip(temp_video_path) as new_video_clip:
                    final_clip = new_video_clip.set_audio(original_clip.audio); final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=4, logger='bar')
            else: os.rename(temp_video_path, output_path)
    except Exception as e: print(f"音声合成中にエラー: {e}"); os.rename(temp_video_path, output_path)
    finally:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
    print(f"\n--- ✅ 分析ビデオ生成完了: {output_path} ---")

def analyze_and_prepare_pzs_data(df_gaze, video_path, baseline_model, screen_w, screen_h):
    with VideoFileClip(video_path) as clip: fps, duration = clip.fps, clip.duration
    if not df_gaze.empty and df_gaze['timestamp'].max() > duration * 1.5: df_gaze['timestamp'] /= 1000.0
    corrected_df = df_gaze[df_gaze['timestamp'] <= duration].copy()
    corrected_df['frame_index'] = (corrected_df['timestamp'] * fps).astype(int)
    corrected_df.interpolate(method='linear', limit_direction='both', inplace=True)
    agg_dict = {col: 'mean' for col in ['pred_x', 'pred_y', 'pupil_radius', 'ear', 'head_pitch', 'head_yaw']}
    df_sync = corrected_df.groupby('frame_index').agg(agg_dict).reset_index()
    total_frames = int(duration * fps)
    df_result = pd.merge(pd.DataFrame({'frame_index': range(total_frames)}), df_sync, on='frame_index', how='left').interpolate(method='linear', limit_direction='both')
    video_cap = cv2.VideoCapture(video_path); luminance_values = []
    for i in tqdm(range(total_frames), desc="視線周辺の輝度を計算中"):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, i); ret, frame = video_cap.read()
        if not ret: luminance_values.append(np.nan); continue
        gaze_x = df_result.loc[i, 'pred_x'] * (video_cap.get(cv2.CAP_PROP_FRAME_WIDTH) / screen_w)
        gaze_y = df_result.loc[i, 'pred_y'] * (video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / screen_h)
        luminance_values.append(get_foveal_luminance(frame, gaze_x, gaze_y))
    video_cap.release()
    df_result['luminance'] = pd.Series(luminance_values).interpolate()
    df_result['predicted_pupil'] = df_result.apply(lambda r: baseline_model.predict(r['luminance'], 'stable') if pd.notna(r['luminance']) else np.nan, axis=1)
    df_result['pupil_residual'] = df_result['pupil_radius'] - df_result['predicted_pupil']
    valid_residuals = df_result['pupil_residual'].dropna()
    mean_res, std_res = valid_residuals.mean(), valid_residuals.std()
    df_result['pzs'] = (df_result['pupil_residual'] - mean_res) / std_res if std_res > 0 else 0
    df_result.rename(columns={'pred_x': 'gaze_x', 'pred_y': 'gaze_y'}, inplace=True)
    df_result['gaze_x_smooth'] = df_result['gaze_x'].rolling(window=max(1, int(fps/5)), min_periods=1, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    df_result['gaze_y_smooth'] = df_result['gaze_y'].rolling(window=max(1, int(fps/5)), min_periods=1, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    print("--- ✅ ハイブリッド剰余法PZS計算完了 ---"); return df_result, corrected_df

def perform_pzs_analysis(gaze_df, calib_df, lum_calib_data, item_name, item_path, screen_w, screen_h):
    print(f"--- 🎬 動画PZS分析開始: {item_name} ---")
    corrected_df = correct_gaze_data(gaze_df.copy(), calib_df, screen_w, screen_h)
    baseline_model = PupilBaselineModel()
    if not baseline_model.train(lum_calib_data): print("--- エラー: 瞳孔ベースラインモデルの訓練失敗。---"); return
    
    df_pzs_analyzed, corrected_df_for_summary = analyze_and_prepare_pzs_data(corrected_df, item_path, baseline_model, screen_w, screen_h)
    
    # ###【バグ修正】タイムスタンプ列を再計算して追加 ###
    with VideoFileClip(item_path) as clip:
        fps = clip.fps
    df_pzs_analyzed['timestamp'] = df_pzs_analyzed['frame_index'] / fps

    timestamp_str = time.strftime('%Y%m%d_%H%M%S')
    output_excel_path = os.path.join(OUTPUTS_PATH, f"result_{os.path.splitext(item_name)[0]}_analysis_{timestamp_str}.xlsx")

    with pd.ExcelWriter(output_excel_path) as writer:
        # Sheet 1: Detailed Data
        df_sheet1 = df_pzs_analyzed[['timestamp', 'frame_index', 'gaze_x_smooth', 'gaze_y_smooth', 'pupil_radius', 'pzs', 'luminance']].copy()
        df_sheet1.rename(columns={
            'timestamp': 'タイムスタンプ(秒)', 'frame_index': 'フレーム番号', 'gaze_x_smooth': '視線位置X', 'gaze_y_smooth': '視線位置Y',
            'pupil_radius': '瞳孔半径', 'pzs': 'Zスコア(PZS)', 'luminance': '輝度値'
        }, inplace=True)
        df_sheet1.to_excel(writer, sheet_name='フレーム別詳細データ', index=False)
        print(f"--- ✅ 詳細データシートを作成しました ---")

        # Sheet 2: Engagement Summary
        on_screen_gaze = corrected_df_for_summary[(corrected_df_for_summary['pred_x'] >= 0) & (corrected_df_for_summary['pred_x'] <= screen_w) & (corrected_df_for_summary['pred_y'] >= 0) & (corrected_df_for_summary['pred_y'] <= screen_h)]
        screen_gaze_rate = len(on_screen_gaze) / len(corrected_df_for_summary) if len(corrected_df_for_summary) > 0 else 0
        
        avg_positive_arousal = df_pzs_analyzed[df_pzs_analyzed['pzs'] > 0]['pzs'].mean()
        if pd.isna(avg_positive_arousal): avg_positive_arousal = 0
        
        cumulative_arousal = df_pzs_analyzed['pzs'].sum()
        peak_arousal = df_pzs_analyzed['pzs'].max()
        engagement_index = avg_positive_arousal * screen_gaze_rate * 100

        summary_data = {
            '指標': ['エンゲージメント指数', '画面注視率', '累積覚醒度', '平均ポジティブ覚醒度', 'ピーク覚醒度'],
            '値': [f"{engagement_index:.2f}", f"{screen_gaze_rate:.2%}", f"{cumulative_arousal:.2f}", f"{avg_positive_arousal:.3f}", f"{peak_arousal:.3f}"],
            '説明': [
                '関心の高さと集中の度合いを総合した指標 (平均ポジティブ覚醒度 × 画面注視率)',
                '視線が画面内にあった時間の割合',
                '動画全体を通しての感情・認知的な反応の総量 (Zスコアの合計)',
                '関心があった際の反応の平均強度 (正のZスコアの平均)',
                '最も強く感情・認知的な反応があった瞬間 (Zスコアの最大値)'
            ]
        }
        df_sheet2 = pd.DataFrame(summary_data)
        df_sheet2.to_excel(writer, sheet_name='エンゲージメントサマリー', index=False)
        print(f"--- ✅ エンゲージメントサマリーシートを作成しました ---")
    
    print(f"--- ✅ 分析結果Excelファイルを保存しました: {output_excel_path} ---")

    # 3. 分析動画の生成
    output_video_path = os.path.join(OUTPUTS_PATH, f"result_{os.path.splitext(item_name)[0]}_pzs_analyzed.mp4")
    create_analyzed_video(item_path, df_pzs_analyzed, output_video_path)
    
    print(f"--- ✅ 動画PZS分析完了: {item_name} ---")


def perform_pre_work_analysis(data):
    print(f"--- 📋 就業前チェック分析開始 ---")
    gaze_df = pd.DataFrame(data['gaze_data'])
    if gaze_df.empty: print("--- 警告: 就業前チェックのデータが空です。---"); return
    
    results = {}
    saccade_data = gaze_df[gaze_df['phase'] == 'saccade'].copy()
    if not saccade_data.empty:
        saccade_data[['gaze_x', 'gaze_y']] = pd.DataFrame(saccade_data['raw_gaze_vector'].tolist(), index=saccade_data.index)
        saccade_data['distance'] = np.sqrt(saccade_data['gaze_x'].diff()**2 + saccade_data['gaze_y'].diff()**2)
        saccade_data['speed'] = saccade_data['distance'] / saccade_data['timestamp'].diff()
        results['saccade_mean_speed'] = saccade_data['speed'].mean(); results['saccade_max_speed'] = saccade_data['speed'].max()
    
    plr_data = gaze_df[gaze_df['phase'] == 'plr'].copy()
    if not plr_data.empty:
        pupil_dark = plr_data[plr_data['stimulus'] == 0]['pupil_radius'].mean()
        pupil_light = plr_data[plr_data['stimulus'] == 1]['pupil_radius'].mean()
        if pd.notna(pupil_dark) and pupil_dark > 0: results['plr_response'] = (pupil_dark - pupil_light) / pupil_dark
        else: results['plr_response'] = np.nan
        
        latencies = []
        stim_changes = plr_data[plr_data['stimulus'].diff() == 1]
        for t_stim in stim_changes['timestamp']:
            response_window = plr_data[(plr_data['timestamp'] > t_stim) & (plr_data['timestamp'] <= t_stim + 1.5)]
            if not response_window.empty:
                pupil_diff = response_window['pupil_radius'].diff()
                if not pupil_diff.dropna().empty:
                    t_response = pupil_diff.idxmin()
                    latencies.append(response_window.loc[t_response, 'timestamp'] - t_stim)
        results['plr_latency'] = np.mean(latencies) if latencies else 0

    fixation_data = gaze_df[gaze_df['phase'] == 'fixation'].copy()
    if not fixation_data.empty:
        fixation_data[['gaze_x', 'gaze_y']] = pd.DataFrame(fixation_data['raw_gaze_vector'].tolist(), index=fixation_data.index)
        results['fixation_stability_score'] = np.mean([fixation_data['gaze_x'].std(), fixation_data['gaze_y'].std()])
    
    historical_files = sorted(glob.glob(os.path.join(OUTPUTS_PATH, "pre_work_check_results_*.json")))
    historical_data = []
    for f in historical_files:
        with open(f, 'r', encoding='utf-8') as file:
            past_data = json.load(file)
            try:
                ts_str = os.path.basename(f).replace("pre_work_check_results_", "").replace(".json", "")
                past_data['timestamp'] = time.mktime(time.strptime(ts_str, '%Y%m%d_%H%M%S'))
                historical_data.append(past_data)
            except ValueError: continue
    
    timestamp_unix = time.time()
    results_to_save = {k: (v if pd.notna(v) else None) for k, v in results.items()}
    timestamp_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp_unix))
    output_filename_json = os.path.join(OUTPUTS_PATH, f"pre_work_check_results_{timestamp_str}.json")
    with open(output_filename_json, 'w', encoding='utf-8') as f: json.dump(results_to_save, f, ensure_ascii=False, indent=4)
    print(f"--- ✅ 就業前チェック分析結果(JSON)を保存しました: {output_filename_json} ---")
    
    current_data_with_ts = results_to_save.copy()
    current_data_with_ts['timestamp'] = timestamp_unix
    historical_data.append(current_data_with_ts)
    historical_df = pd.DataFrame(historical_data).sort_values(by='timestamp').reset_index(drop=True)

    try:
        output_filename_png = os.path.join(OUTPUTS_PATH, f"pre_work_check_dashboard_{timestamp_str}.png")
        generate_pre_work_dashboard(gaze_df, results, output_filename_png, historical_df)
    except Exception as e:
        print(f"--- 🚨 ダッシュボード生成中にエラーが発生しました ---"); print(f"エラー内容: {e}"); traceback.print_exc()

# =============================================================================
# SECTION 4: Flask Webサーバー
# =============================================================================
def run_analysis_in_background(data):
    try:
        gaze_df = pd.DataFrame(data['gaze_data']); item_name = data['item_name']
        analysis_type = data.get('analysis_type', 'standard')
        if analysis_type == 'pre_work_check': perform_pre_work_analysis(data); return
        gaze_calib_df = pd.DataFrame(data['gaze_calibration_data']); lum_calib_data = data.get('luminance_calibration_data', [])
        is_video, screen_w, screen_h = data['is_video'], data['screen_w'], data['screen_h']
        item_path = data.get('item_path')
        if is_video:
            if item_path and os.path.exists(item_path): perform_pzs_analysis(gaze_df, gaze_calib_df, lum_calib_data, item_name, item_path, screen_w, screen_h)
        else:
             if item_path: perform_heatmap_analysis(gaze_df, gaze_calib_df, lum_calib_data, item_name, item_path, screen_w, screen_h)
        gc.collect(); print(f"--- ✅ 分析完了: {item_name} ---")
    except Exception: traceback.print_exc()

@app.route('/upload', methods=['POST'])
def upload_data():
    try:
        data = request.json
        thread = threading.Thread(target=run_analysis_in_background, args=(data,))
        thread.start()
        return jsonify({"message": "分析リクエストを受理しました。"}), 202
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

