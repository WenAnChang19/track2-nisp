# BMI三模型：人口資訊不懲罰與共同交叉驗證

日期：2026-10-04。本輪在本機CPU完成；沒有SSH、國網作業、新權重下載或GPU使用。只處理BMI三模型與必要診斷，未啟動身高、選層、bootstrap或分組。

## 執行前固定的規則

使用已保存的345人speaker bundle，只對原train168＋validation42人的210人建立development資料。test135人只核對數目，不解析BMI／年齡／性別進入模型。

設定見`configs/bmi_partial_ridge_cv_v2.json`。v1是執行前草案；審查後、任何真實模型執行前，v2明列同折OOF性別診斷、收斂與rank防呆及鎖檔要求，沒有根據結果修改設定。Runner要求v2原始bytes SHA完全一致。`protocol_lock.json`在第一次擬合前保存設定、來源特徵、四個實際使用source檔的SHA、library versions、ordered IDs與完整分折名單；後續數值檔引用同一鎖檔hash。

- 依性別分層的5折×5次重複，seed42；210人為女性76／男性134。
- 各折人口設計為截距、年齡與性別（Female0／Male1），同一個OLS不受懲罰。
- 語音特徵只用training fold均值／標準差標準化；BMI與標準化後語音都對training fold人口資訊殘差化。Ridge只懲罰語音，OOF預測加回人口部分；OOF BMI只算誤差，不參與擬合。
- MFCC26維、WavLM固定第12層768維。每模型用相同16個alpha（10^-3～10^12），按25fold平均MAE選；與最佳分數差不超過10^-6時選最大alpha。
- 25fold是相依的開發比較，SD只作描述，不用其SE作嚴格推論。同一CV既選alpha又報分數會有樂觀偏差。

## 結果

| 模型 | 選定alpha | 平均CV MAE（kg/m²） | 與人口基準的MAE差 | 較好／持平／較差折數 |
|---|---:|---:|---:|---:|
| 人口資訊 OLS | 不懲罰 | 2.940 | +0.000 | — |
| MFCC＋人口資訊 | 1e+12 | 2.940 | +0.000 | 0／25／0 |
| WavLM 第12層＋人口資訊 | 1000 | 2.854 | -0.086 | 16／0／9 |

WavLM第12層的開發MAE比人口OLS少約0.086 kg/m²，25折中16折較好、9折較差。這是小幅開發增益，尚不能宣稱test有改善、具統計顯著性或可預測體脂率。

MFCC選到alpha=10^12，幾乎完全退回人口OLS；本輪未顯示MFCC在人口資訊之外的額外效用。

## 嵌套、置換與性別診斷

50個large-alpha檢查全部通過：每一fold的兩個語音模型，在alpha=10^12時，預測和MAE都回到同一人口OLS基準，最大預測差為6.57e-08，低於預先固定的10^-6容許值。

99次同性別內BMI置換使用同一CV與相同選alpha程序，各次都重新擬合人口與目標係數，只重用不依賴y的特徵分解。MFCC有38次、SSL有40次出現超過10^-6的偶然改善，顯示不能把「單次打亂後勝過基準」直接判為洩漏。SSL置換的最大改善約0.046 kg/m²，99次沒有達到原始資料的0.086改善；這只是描述性的診斷結果，不是正式p值、不證明無洩漏。置換破壞age-BMI關聯，不能當作控制年齡後的嚴格null。

純語音性別分類使用共同OOF folds、training-only scaler、固定LogisticRegression（C=1），沒有輸入年齡或性別；收斂警告會中止診斷。

| 輸入 | 平均OOF accuracy | 平均balanced accuracy |
|---|---:|---:|
| MFCC | 98.19% | 98.29% |
| WavLM第12層 | 97.33% | 97.91% |

多數類別基準accuracy約63.81%、balanced accuracy50%。這支持語音特徵含可用的性別訊號，沒有看到明顯診斷異常；性別分類成功本身不能排除同一性別內的錯配，亦不保證BMI增益。

## 獨立驗證

1,050筆OOF預測（210人×5次）重新計算的75個fold MAE／RMSE／R²，與輸出在10^-12容許誤差內一致。25fold平均、配對勝負及alpha選擇規則均核對通過；每人每repeat恰有一筆OOF，沒有test預測或指標。所有數值artifact的lock hash一致。詳細見`independent-verification.json`。

Synthetic tests另驗證與獨立partial-penalty解等價、training-only scaling、改變test污染值不影響選alpha或OOF數值、維度防呆、共同分折與輸出範圍。舊新全套38項通過；最後兩項防呆修改後，CV專屬11項再次通過。

## 新舊結果的關係與後續

第一批的3.237／3.374／3.313是單一42人validation與普通共同懲罰Ridge，本輪是210人重複CV與人口不懲罰。兩項設計同時改變，不能把新舊數字直接當成模型提升，也不能斷言原本落後完全由人口懲罰造成。第一批資料與表圖保留，沒有覆寫。

本輪乾淨交付到此停止。下一步按已同意順序補平均值／中位數及聲學單獨基準，再做身高與預先固定規則的12層比較。模型／層／alpha／前處理全部鎖定後，再以210人重訓並作一次正式test評估批次；bootstrap及分組只在預先指定的結果上分析，不回頭用test挑設定。

## 重現

從track2-nisp目錄執行：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-tests/bin/python evaluate_nisp_cv.py \
  --features results/stage1-20261004T092357Z/features \
  --config configs/bmi_partial_ridge_cv_v2.json \
  --output results/NEW_CV_OUTPUT
```

輸出目錄必須不存在；加`--plan`可只核對資料與分折而不跑模型。主執行紀錄在`reports/bmi-cv-v2-execution.log`。
