# 第一批 NISP BMI 實驗結果

2026-10-04 完成。這是語音身體資訊方法的 BMI 開發實驗；NISP 沒有直接體脂率標籤，不能將本表稱為體脂率預測成績。

## 完成範圍

- 14,691 段英文錄音、345 位說話者。保留既有 train 168／validation 42／test 135 人切分。
- 每段最多前10秒，從實際44.1或48 kHz重採樣為16 kHz。frozen WavLM Base+保存全部12個Transformer層的時間平均，每層768維；同一段音訊提取13個MFCC的mean/std，共26維。
- 每位說話者以錄音等權平均，資料來源、ID排序、切分、重採樣長度、有限數值與平均值通過獨立核對。國網保存utterance特徵及checkpoint，本機保存speaker特徵及表圖。
- StandardScaler與Ridge僅在train擬合；各模型用相同alpha網格按validation MAE選擇。SSL固定第12層。沒有執行test預測、test指標、bootstrap、分組或選層分析。

## Validation結果（42人）

| 模型 | Alpha | MAE（kg/m²） | RMSE（kg/m²） | R² |
|---|---:|---:|---:|---:|
| 模型2：年齡＋性別 | 0.001 | 3.237 | 4.020 | 0.047 |
| 模型5：MFCC＋年齡＋性別 | 1000 | 3.374 | 4.092 | 0.012 |
| 模型6：WavLM第12層＋年齡＋性別 | 10000 | 3.313 | 4.028 | 0.043 |

三個模型的R²接近0，散點圖顯示預測集中在BMI約23附近，極端BMI常被拉向平均。第一批加入MFCC或WavLM第12層後，MAE均未優於年齡＋性別。這描述目前設定的validation結果，尚未做不確定性分析，不能推論語音完全沒有相關資訊，也不能當作最終test成績。

## 執行與核對

從42筆逐人validation預測，以NumPy獨立重算三模型MAE／RMSE／R²，與JSON指標吻合；表圖只包含validation資料，PNG視覺檢查通過。

28項測試通過；100段train-only pilot提取與12層對齊通過，第二次100段全部由checkpoint重用，沒有重新forward。

- 首次pilot job495925：FAILED，9秒，GPU step看不到CUDA，未做模型推論。
- 新啟動器明確指定step GPU並記錄診斷；pilot job496518：COMPLETED，38秒，H200節點25a-hgpn003。
- 全資料job496531：COMPLETED，8分56秒，提取step6分4秒、獨立核對step2分49秒。CUDA問題在本次未重現，尚不能斷定首次失敗的底層原因。
- 發現核對程式反覆解壓NPZ來源索引；本機後續版本已改為一次載入，語法與小fixture核對通過。此次國網執行仍使用保留的原始快照，最終核對成功，沒有重跑特徵。

第一批三個配置GPU工作合計583秒；依用戶提供的30元/GPU-hour按秒比例約4.86元。這是時間比例估算，實際扣款與最低計費單位未核實。先前五段smoke的13秒不包含在此數字。

## 檔案與重現

國網來源快照與全部特徵：`data/nisp/runs/stage1-20261004T092357Z`。

本機同目錄下：`features/`含speaker bundle與hash核對、`validation/`含三模型、protocol、逐人預測與指標、`figures/`含PNG/PDF散點圖及比較表。原始12層utterance特徵留在國網，後續層比較不需重跑GPU。

在track2-nisp目錄用`.venv-tests/bin/python evaluate_nisp.py --features results/stage1-20261004T092357Z/features --output NEW_OUTPUT_DIR`重現CPU評估。程式拒絕覆蓋既有輸出；繪圖使用`render_nisp_stage1.py --evaluation NEW_OUTPUT_DIR --output NEW_FIGURE_DIR`。

## 第二批仍待執行

按用戶分批規劃，下一批才補平均值與傳統聲學單獨模型、bootstrap、分組分析、validation選層及身高目標。先完成並鎖定train/validation設定，再以train+validation重訓並正式評估test。BMI結果不能替代仍在等待回覆的體脂率資料集。
