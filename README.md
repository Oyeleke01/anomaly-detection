# Industrial Anomaly Detection (MVTec AD): EfficientNet-B2 + FastAPI

Binary defect detection (normal vs defective) on three MVTec AD categories
(bottle, screw, carpet). One EfficientNet-B2 transfer-learning model per
category, trained on AWS SageMaker (Tesla T4) and served with FastAPI.

## Results (held-out test sets, single evaluation)
| Category | Test n | Recall | Precision | F1 | AUC | Missed defects |
|---|---|---|---|---|---|---|
| Bottle | 34 | 88.5% | 100% | 93.9% | 1.000 | 3 of 26 |
| Screw | 64 | 59.6% | 100% | 74.7% | 0.973 | 19 of 47 |
| Carpet | 48 | 97.3% | 92.3% | 94.7% | 0.961 | 1 of 37 |

## Data leakage found and fixed
An earlier version trained on test-folder defects and was evaluated on the same
folder (screw ~99%). All models were retrained on a stratified
train/validation/test split with an overlap assertion. The numbers above are
the corrected ones.

## Limitations
- Small test sets (34/64/48 images), so results are noisy
- Not the standard MVTec protocol (trains on 40% of test-folder images, including defects),
  so results are not comparable with published anomaly-detection scores
- Screw recall is 0.60, so it is not production-ready
- Academic project, not validated for production use

## API
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000
    curl -X POST -F "file=@image.png" http://localhost:8000/predict/bottle

Endpoints: GET /health, POST /predict/{bottle|screw|carpet}

## Stack
PyTorch, torchvision, EfficientNet-B2, FastAPI, AWS SageMaker, S3, EC2

## Author
Paul Oyeleke, github.com/Oyeleke01
