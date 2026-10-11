# Track 2: NISP speech and body-information experiments

以凍結 WavLM 與 MFCC 特徵檢驗語音是否提供年齡、性別之外的 BMI／身高預測資訊。訓練的是 Ridge 回歸，沒有微調 WavLM。本儲存庫是研究專案的整理副本，沒有錄音、逐人身體資料、特徵、權重或 SSH 設定。

## 結果

正式 test 135 人：BMI 人口 OLS MAE 2.733386、選定第5層 WavLM＋人口 2.407814 kg/m²；身高人口 5.135488、選定第3層 5.042051 cm。BMI 改善通過預先固定的區間門檻，身高尚未確認額外增益。BMI 不是體脂率，結果不是因果證明。

[完整實驗說明](docs/EXPERIMENT_EXPLANATION.md) · [正式結果與圖表](results_summary/official-test-scores-v2-20261005/REPORT.md) · [研究規劃](EXPERIMENT_PLAN.md)

## 安裝

CPU 開發／回歸的歷史環境為 Python 3.12；先建立自己的虛擬環境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
```

若需要重新提取 WavLM 特徵，另依硬體安裝相容的 PyTorch／torchaudio，再安裝 `requirements-features.txt`。歷史GPU環境為 Python 3.11、torch／torchaudio 2.10.0、transformers 4.57.6；CPU 回歸不需要 GPU。

## 程式流程

| 程式 | 工作 |
|---|---|
| prepare_nisp.py | 建立英文錄音 manifest 與說話者切分 |
| download_wavlm.py、extract_nisp.py、nisp_features.py | 固定 WavLM revision、提取特徵及每人平均 |
| cv_partial_ridge.py、audio_ridge_cv.py | 人口不懲罰的雙殘差 Ridge、純語音 Ridge |
| evaluate_nisp.py | 第一批歷史 validation，亦提供後續相依工具 |
| evaluate_nisp_cv.py | BMI 開發 CV 與診斷 |
| evaluate_nisp_baselines_height.py | 常數／純語音基準及身高 |
| evaluate_nisp_layers.py | 開發資料12層比較與選層 |
| evaluate_nisp_selection_permutation.py | 完整選擇過程置換 |
| nisp_final_inputs.py、evaluate_nisp_test.py | 分離目標，最終重訓、先預測後計分 |
| render_nisp_stage1.py、render_nisp_official_test.py | 從已保存結果繪圖 |

上傳副本已在本機以合成資料驗證，72 項單元測試全部通過；未重跑真實實驗。

`configs/` 保存原固定設定；`tests/` 是合成資料單元測試，不需要真實 NISP 資料。

```bash
python -m unittest discover -s tests -v
```

## 資料與重現界線

請從 [NISP 官方資料集](https://github.com/iiscleap/NISP-Dataset) 取得資料並遵循原授權，模型由 [WavLM Base+](https://huggingface.co/microsoft/wavlm-base-plus) 取得。預設資料位置為 `data/nisp`，也可指定 `--root`；下載權重另需明確的 `--approved-download` 旗標。

本儲存庫提供程式與集計摘要，**不是只需 clone 就能重現全部歷史結果的資料包**。開發與最終 runner 會驗證原特徵指紋、fold 與前輪結果 hash；這些原始資料、逐人預測與完整 protocol locks 未分發。完整歷史重現需要研究者提供對應的私有 artifact bundle。不要刪除 hash gate 或拿已使用的 test 重新選參數。

上傳副本只把個人資料根目錄改成通用 `data/nisp`，因此部分 source hash 與原鎖定快照不同；原實驗檔案完全保留。新結果應另立設定與鎖定，不能覆寫歷史紀錄。先前討論的母語留一群評估尚未實作。

## 授權

本儲存庫不重新分發 NISP 資料或 WavLM 權重；它們各自的授權不因本儲存庫而改變。本次未替專案程式新增開源授權，程式授權待專案擁有人選定。
