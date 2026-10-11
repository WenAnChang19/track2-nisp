# 本機第二輪：新增基準與身高

已完成兩個目標、七個模型的固定第12層開發CV。原BMI三模型的alpha、25-fold平均MAE及1,050筆OOF預測與上一輪完全一致；新增結果沒有改動原實驗。

## 目前結果

| 模型 | BMI MAE（kg/m²） | 身高 MAE（cm） |
|---|---:|---:|
| 訓練折平均值 | 3.018 | 7.691 |
| 訓練折中位數 | 2.981 | 7.701 |
| 年齡＋性別（OLS） | 2.940 | 5.741 |
| MFCC 單獨 | 3.016 | 5.631 |
| WavLM 第12層單獨 | 2.835 | 5.547 |
| MFCC＋年齡＋性別 | 2.940 | 5.457 |
| WavLM 第12層＋年齡＋性別 | 2.854 | 5.596 |

BMI：MFCC單獨MAE 3.016，接近平均值3.018；MFCC加人口資訊選alpha=10¹²，退回人口基準2.940。WavLM第12層單獨為2.835，加人口資訊為2.854，後者相對人口基準少0.086 kg/m²（25折中16勝9負）。這是小幅開發增益，仍待正式test確認。

身高：人口基準5.741 cm，MFCC加人口資訊5.457 cm，減少0.284 cm（22勝3負）；WavLM加人口資訊5.596 cm，減少0.144 cm（17勝8負）。固定第12層下，傳統MFCC加人口資訊的開發分數較低。身高是輔助診斷，這些改善不能證明生理因果，也不是流程正確性的唯一證據。

純語音可能包含性別等人口資訊；純語音優於常數基準不能直接解讀為人口資訊以外的身體訊號。主要比較仍是同目標的「語音＋人口資訊」對「人口資訊」。WavLM單獨略優於加入人口資訊不是程式必然錯誤：模型嵌套只確保存在退回人口基準的設定，不保證有限樣本的CV MAE單調改善；擬合使用平方損失，選參數使用MAE。

## 固定流程

- 210位說話者、一人一列，沿用上一輪完整25組training／validation ID。每折168人擬合、42人評分。
- 兩目標各自選各語音模型的alpha；同一10⁻³至10¹²網格，平均fold MAE差在10⁻⁶內時取最大alpha。沒有跨目標交換設定。
- 平均值與中位數只由各training折標籤計算。
- 純語音：training折標準化、普通Ridge、截距不懲罰，輸入不含年齡／性別。
- 語音＋人口資訊：training折的OLS年齡／性別不懲罰，目標與語音雙殘差化，僅語音係數受Ridge懲罰。
- 身高、體重、BMI都不作模型輸入。test僅檢查135人的split數量，不解析其人口數值或目標。
- 特徵仍是凍結WavLM前10秒提取後的每人平均；MFCC為13係數mean/std共26維。未重提特徵，未使用國網或GPU。

## 檢查與可重現性

每次fit前保存原始設定檔、完整fold IDs、特徵指紋、實際計算source hashes、套件版本及前輪結果hash。既有輸出目錄不可覆寫。

BMI的三模型完全重現；200項大alpha檢查全部通過：純語音退回各training折平均值，語音＋人口資訊退回同折人口OLS，預測及MAE差均小於10⁻⁶。新的9項focused tests通過，涵蓋獨立scikit-learn oracle、常數模型與heldout隔離等。實際OOF與CSV指標的獨立核對見`independent-verification.json`，獨立重算已通過（status: ok）。

共有2,100筆OOF列（兩目標×210人×五次重複）、350個模型fold指標、128列alpha搜尋、14列比較結果。逐人預測未先平均五次重複來報告指標，避免與上述25-fold MAE混淆。

設定：`../../configs/baselines_height_cv_v1.json`；原始SHA-256為`0f2a2710d19718548ad73360cd0249c7c72e446038701e7b737d9a53093d6448`。本輪protocol lock SHA-256為`09b1e90ce66182f8d143d816e3fa88211ca645171864e974690bffe69069e357`。

從track2-nisp目錄重現，另取不存在的新輸出目錄：

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-tests/bin/python evaluate_nisp_baselines_height.py \
  --features results/stage1-20261004T092357Z/features \
  --config configs/baselines_height_cv_v1.json \
  --prior-cv results/bmi-cv-v2-20261004 \
  --output results/baselines-height-cv-reproduction
```

## 結果檔與後續

`comparison.md`包含完整MAE、alpha與成對差異表；`cv_comparison.png`／`.pdf`為兩目標圖；`cv_metrics.csv`、`cv_fold_metrics.csv`、`cv_alpha_search.csv`及`cv_predictions.csv`保留計算數據；`bmi_reproduction.json`、`nesting_checks.json`及`protocol_lock.json`保留驗證紀錄。

下一階段在看層結果前另存12層比較規則，沿用這210人的共同CV，分別選層及alpha。選層CV曲線用於設定選擇，不把最低點當作最終成績；不串接12層。鎖定最終設定後才以210人擬合並一次評估test，之後做成對bootstrap與描述性分組。這一輪沒有做層選擇、正式test、bootstrap、分組或最終210人擬合。

NISP BMI是由身高與體重算出，不能當成體脂率標籤；真正體脂率資料仍待回覆。第一批單一validation與舊共同Ridge懲罰的結果保留為歷史，不與本輪CV數字直接解讀成模型改善幅度。
