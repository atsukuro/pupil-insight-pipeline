\# Pupil Insight Pipeline



\## 概要

瞳孔径・視線・瞬きのリアルタイム分析を行うPythonベースのパイプラインです。現在はWebcam＋MediaPipeによる動作を確認済みで、Tobii Eye Tracker 5 SDKとの統合も視野に入れて開発を進めています。教育、UX、医療、心理領域での応用を想定し、SDK統合からデバイス制御、データ取得、可視化までを一貫して設計しています。



\## 現在の実装状況

\- ✅ Webcam＋MediaPipeによる瞳孔・瞬き検出は動作確認済み

\- 🔄 Tobii Eye Tracker 5 SDKとの統合は準備中（SDK連絡待ち）

\- 🧪 Tobiiデバイスによる視線・瞳孔径取得は今後の実装予定



\## 特徴

\- MediaPipeによる顔・目領域の検出と瞳孔解析

\- 瞬き検出とタイムスタンプ記録

\- Pythonによる軽量な実装

\- WebSocket/MQTTによる外部連携（オプション）

\- 教育・UX・医療・心理分野への応用可能性



\## 技術スタック

\- Python 3.9+

\- OpenCV

\- MediaPipe

\- NumPy / Pandas

\- WebSocket / MQTT（任意）



\## デモ

!\[demo](demo.gif)  

※Webcamによる瞳孔・瞬き検出のリアルタイム表示



\## 応用例

\- \*\*教育\*\*：学習時の集中度・理解度の可視化

\- \*\*UXリサーチ\*\*：UIへの視線集中と瞳孔反応の定量化

\- \*\*医療・介護\*\*：認知症・精神状態のモニタリング

\- \*\*マーケティング\*\*：広告への反応を瞳孔で測定

\- \*\*人材評価\*\*：面接時の緊張・反応の分析



\## 使用方法

```bash

git clone https://github.com/yourname/pupil-insight-pipeline.git

cd pupil-insight-pipeline

pip install -r requirements.txt

python main.py

