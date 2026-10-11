# 12層比較完成：BMI與身高

固定相同210位development說話者與既有25-fold CV，完成WavLM全部12個Transformer層的比較；每層仍只用768維，並與不懲罰的年齡／性別OLS組合。正式test135人未評估。

## 選定設定

| 目標 | 人口基準MAE | 第12層＋人口MAE | 選定層＋人口MAE | 選定alpha |
|---|---:|---:|---:|---:|
| BMI（kg/m²） | 2.940 | 2.854 | 第5層：2.699 | 1000 |
| 身高（cm） | 5.741 | 5.596 | 第3層：5.136 | 1000 |

BMI選定模型對人口基準減少0.241 kg/m²，25-fold中22勝3負；身高減少0.604 cm，23勝2負。身高同時低於前輪MFCC＋人口基準5.457 cm；BMI MFCC＋人口基準是2.940。

原始最低點分別為BMI第4層2.655、身高第2層5.118。執行前已固定近似平手帶及優先靠近第6層的規則，因此最後選第5／3層，不把最低點直接當作選定模型的成績。完整候選、alpha及數字見`comparison.md`。結果顯示，在這份資料與現有提取方式下，只看第12層會漏掉較低誤差的候選層；它不支持「12層特徵太少」的解讀。

## 規則與解讀

各target×layer使用同一alpha網格10⁻³～10¹²，按25個fold平均MAE選alpha，差在10⁻⁶內時取較大alpha。層選擇先找各層已選alpha的最低MAE，納入距最低點不超過BMI0.05 kg/m²、身高0.10 cm的候選，再選最接近第6層者，同距離取較低層。這是預先固定的實務選擇帶，不是統計等效檢定、信賴區間或one-standard-error rule；圖上橘色區域同理。

第6層只提供確定性的平手偏好，不是假定中層有生理優勢。各折重疊，fold SD與勝負次數只作描述，不由25折推算獨立樣本標準誤或顯著性。曲線用於設定選擇，同一CV選alpha與layer會有樂觀偏差，最低點與選定分數都不是正式泛化成績。

改善也不能證明聲音與身體指標的因果連結。身高是輔助診斷；BMI由身高與體重算出，不能當作實測體脂率。真正體脂率資料仍待申請回覆。

## 隔離與核對

- 所有標準化、人口殘差化與回歸均在各training折168人內擬合，42人validation只套用。12層不串接、沒有新增維度或重新提取特徵。
- 使用原先保存的完整25組fold IDs，並驗證特徵指紋、ordered development IDs及前輪檔案hash。
- 每次擬合前保存精確config bytes、7個實際計算source hashes、前輪reference hashes、完整fold IDs與套件版本；拒絕覆寫既有run。
- 先做第12層，必須完整重現兩目標的SSL＋人口與人口OLS：每個target1050筆keyed OOF預測、mean-fold MAE及SSL alpha均一致，差值為零。通過後才執行其他11層。
- 600項大alpha檢查（兩目標×12層×25折）全部通過，必須退化為同fold人口OLS，預測與MAE差小於10⁻⁶，通過後才寫出結果指標。
- 新8項focused tests通過，涵蓋層索引、獨立回歸oracle、alpha、近似平手、重現與nesting阻擋及拒覆寫。
- 額外獨立joint-penalty normal equations以兩目標、高維少樣本、常數欄和偏移的validation檢查6種設定，最大預測差3.12×10⁻⁹；validation目標改變不影響預測，test數值污染不進development資料。
- 25,200筆OOF經獨立scikit-learn指標重算，600fold MAE／RMSE／R²／人口指標及24摘要均一致；384alpha搜尋、兩組層選擇、OOF IDs與所有鎖定hash也核對一致。另有read-only peer audit交叉核對，沒有新擬合或讀取test目標。

所有計算在本機CPU完成，沒有SSH或國網GPU工作。test僅檢查135人的split數量，不解析其目標或人口數值，也未建立最終210人模型。

## 輸出與重現

表與曲線：`comparison.md`、`layer_cv_curves.png`／`.pdf`。選定設定：`selected_layer_settings.json`、`layer_selection.json`。數據：`layer_cv_metrics.csv`（24列）、`layer_cv_fold_metrics.csv`（600列）、`layer_cv_alpha_search.csv`（384列）、`layer_cv_predictions.csv`（25,200列）。驗證：`layer12_reproduction.json`、`nesting_checks.json`、`independent-verification.json`、`peer-readonly-audit.json`及`protocol_lock.json`。

正式設定檔為`../../configs/layer_selection_cv_v2.json`，SHA-256 `ca52633b8ca7dd6f7be7e0887378a4b8fdff3637b080f8ca564764b576764654`；protocol lock SHA-256 `e4beb78e4d74fe7e7d0af897a1bda71ee4f08b4aa89fff249cc402c97905f52f`。v2在任何新層擬合前補上直接人口OLS重現檢查，選層規則未改動。

從track2-nisp目錄重現，指定不存在的輸出目錄：

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-tests/bin/python evaluate_nisp_layers.py \
  --features results/stage1-20261004T092357Z/features \
  --config configs/layer_selection_cv_v2.json \
  --prior-cv results/bmi-cv-v2-20261004 \
  --layer12-reference results/baselines-height-cv-v1-20261004 \
  --output results/layers-cv-v2-reproduction
```

## 下一階段

依已選定BMI第5層、身高第3層及alpha1000，先固定正式test的比較模型、主要指標、bootstrap與分組描述規則，再以210人擬合，一次評估官方test135人。其後做同一test受試者的成對bootstrap與描述性分組，不用test結果再選層或調alpha。當前交付停在完成開發設定選擇，沒有執行test、bootstrap、分組或最終擬合。
