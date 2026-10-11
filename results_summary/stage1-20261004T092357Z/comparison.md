# First-batch BMI validation

| Model | Alpha | MAE (kg/m²) | RMSE (kg/m²) | R² | n |
|---|---:|---:|---:|---:|---:|
| 2: Age + sex | 0.001 | 3.237 | 4.020 | 0.047 | 42 |
| 5: MFCC + age + sex | 1000 | 3.374 | 4.092 | 0.012 | 42 |
| 6: WavLM layer 12 + age + sex | 10000 | 3.313 | 4.028 | 0.043 | 42 |

Models were fitted on train and selected on validation. These are development scores, not final test results.
