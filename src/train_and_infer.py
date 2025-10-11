"""
train_and_infer.py
Single-file pipeline for Amazon ML Pricing Challenge baseline.

Outputs:
 - dataset/test_out.csv  (matches sample_test_out.csv format)
 - Documentation_submission.md (one-page doc draft)
 - artifacts: oof_preds.npy, model.joblib, embeddings npz
Notes:
 - By default images are OFF (faster). Enable with --use_images True (may need GPU).
 - Uses only provided dataset files (no external price lookups).
"""

import os, sys, argparse, re, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from joblib import dump, load
from tqdm.auto import tqdm

# optional heavy imports
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import torch
    import torchvision.transforms as T
    from torchvision import models
    from PIL import Image
    import requests
except Exception:
    torch = None

# -------------------------
# Utility functions
# -------------------------
def smape(y_true, y_pred, eps=1e-9):
    num = np.abs(y_true - y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, eps)
    return np.mean(num / denom) * 100.0

def extract_pack_qty(text):
    s = str(text).lower()
    # common patterns
    m = re.search(r'(?:pack|packs|set|set of|box of|packet of)\s*(?:of\s*)?(\d{1,4})', s)
    if m:
        try:
            return int(m.group(1))
        except:
            return 1
    m2 = re.search(r'(\d{1,3})\s*(?:count|pcs|pieces|pieces\))', s)
    if m2:
        try:
            return int(m2.group(1))
        except:
            return 1
    # fallback: look for "xN" style like "3 x"
    m3 = re.search(r'(\d{1,3})\s*[xX]\s*\d*', s)
    if m3:
        try:
            return int(m3.group(1))
        except:
            return 1
    return 1

def extract_unit_flags(text):
    s = str(text).lower()
    return {
        'has_gb': int('gb' in s),
        'has_tb': int('tb' in s),
        'has_mp': int('mp' in s),
        'has_mah': int('mah' in s),
        'has_ml': int('ml' in s),
        'has_kg': int('kg' in s)
    }

# simple image downloader + embedder (ResNet50)
def get_resnet_model(device='cpu'):
    if torch is None:
        return None, None
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    model = model.to(device)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return model, transform

def embed_image_url(url, model, transform, device='cpu', timeout=6):
    # returns np.array or zeros if fail
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(resp.raw).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            v = model(x).cpu().numpy().squeeze()
        return v
    except Exception as e:
        return None

# -------------------------
# Pipeline main
# -------------------------
def main(args):
    data_dir = Path('dataset')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    sample_out_path = data_dir / 'sample_test_out.csv'  # formatting reference

    assert train_path.exists(), "dataset/train.csv not found"
    assert test_path.exists(), "dataset/test.csv not found"
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train {len(train):,} rows, Test {len(test):,} rows")

    # quick EDA (prints)
    print("Price stats:", train.price.describe().to_dict())

    # basic text features
    for df in (train, test):
        df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
        df['catalog_len'] = df['catalog_content'].str.len()
        df['num_words'] = df['catalog_content'].str.split().str.len().fillna(0).astype(int)
        df['pack_qty'] = df['catalog_content'].apply(extract_pack_qty)
        unit_flags = df['catalog_content'].apply(extract_unit_flags).apply(pd.Series)
        for c in unit_flags.columns:
            df[c] = unit_flags[c]

    # create price bins on train for stratified CV
    train['price_bin'] = pd.qcut(np.log1p(train['price']), q=10, duplicates='drop', labels=False)

    # choose modeling inputs
    numeric_cols = ['catalog_len', 'num_words', 'pack_qty', 'has_gb','has_tb','has_mp','has_mah','has_ml','has_kg']
    X_train_num = train[numeric_cols].values
    X_test_num  = test[numeric_cols].values

    # TF-IDF baseline
    if args.mode in ('quick','full'):
        print("TF-IDF vectorizing...")
        tfv = TfidfVectorizer(max_features=40000, ngram_range=(1,2))
        X_tfidf_train = tfv.fit_transform(train['catalog_content'])
        X_tfidf_test  = tfv.transform(test['catalog_content'])

        # reduce tfidf with SVD
        svd = TruncatedSVD(n_components=200, random_state=42)
        X_tfidf_train_svd = svd.fit_transform(X_tfidf_train)
        X_tfidf_test_svd  = svd.transform(X_tfidf_test)
    else:
        X_tfidf_train_svd = np.zeros((len(train), 0))
        X_tfidf_test_svd  = np.zeros((len(test), 0))

    # optional text embeddings
    if args.use_text_emb:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        emb_model_name = args.text_emb_model
        print("Computing text embeddings with", emb_model_name)
        s_model = SentenceTransformer(emb_model_name)
        # encode in batches
        emb_train = s_model.encode(train['catalog_content'].tolist(), show_progress_bar=True, batch_size=256)
        emb_test  = s_model.encode(test['catalog_content'].tolist(), show_progress_bar=True, batch_size=256)
        # reduce
        pca_text = PCA(n_components=128, random_state=42)
        emb_train_p = pca_text.fit_transform(emb_train)
        emb_test_p  = pca_text.transform(emb_test)
    else:
        emb_train_p = np.zeros((len(train), 0))
        emb_test_p  = np.zeros((len(test), 0))

    # optional image embeddings
    if args.use_images:
        if torch is None:
            print("Torch not available - skipping image embeddings")
            emb_img_train_p = np.zeros((len(train),0))
            emb_img_test_p = np.zeros((len(test),0))
        else:
            device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
            print("Image embedding model on device:", device)
            resnet, transform = get_resnet_model(device=device)
            # load or compute images
            img_embed_path_train = Path('image_emb_train.npy')
            img_embed_path_test  = Path('image_emb_test.npy')
            def compute_embeds(df, path, src='train'):
                if path.exists():
                    print(f"Loading cached image embeddings {path}")
                    arr = np.load(path, allow_pickle=False)
                    return arr
                embeds = []
                for url in tqdm(df['image_link'].fillna('').astype(str).tolist(), desc=f"img_embed_{src}"):
                    vec = embed_image_url(url, resnet, transform, device=device)
                    if vec is None:
                        vec = np.zeros(resnet.fc.in_features if hasattr(resnet,'fc') else 2048)
                    embeds.append(vec)
                arr = np.vstack(embeds).astype(np.float32)
                np.save(path, arr)
                return arr
            emb_img_train = compute_embeds(train, img_embed_path_train, 'train')
            emb_img_test  = compute_embeds(test, img_embed_path_test, 'test')
            # PCA
            pca_img = PCA(n_components=128, random_state=42)
            emb_img_train_p = pca_img.fit_transform(emb_img_train)
            emb_img_test_p  = pca_img.transform(emb_img_test)
    else:
        emb_img_train_p = np.zeros((len(train), 0))
        emb_img_test_p  = np.zeros((len(test), 0))

    # assemble final train/test feature matrices
    X_tr = np.hstack([X_train_num, X_tfidf_train_svd, emb_train_p, emb_img_train_p])
    X_te = np.hstack([X_test_num,  X_tfidf_test_svd,  emb_test_p,  emb_img_test_p])

    print("Final feature shapes:", X_tr.shape, X_te.shape)

    # standardize numeric-ish features (not sparse)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # target
    y = np.log1p(train['price'].values)

    # 5-fold stratified by price_bin
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    preds_test = np.zeros(len(test))
    models = []
    for fold, (tr_idx, val_idx) in enumerate(folds.split(X_tr, train['price_bin'])):
        print("Fold", fold+1)
        X_tr_f, X_val_f = X_tr[tr_idx], X_tr[val_idx]
        y_tr_f, y_val_f = y[tr_idx], y[val_idx]
        lgbm = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=256,
            subsample=0.8,
            colsample_bytree=0.6,
            random_state=42
        )
        lgbm.fit(X_tr_f, y_tr_f,
                 eval_set=[(X_val_f, y_val_f)],
                 early_stopping_rounds=100,
                 verbose=100)
        oof[val_idx] = lgbm.predict(X_val_f)
        preds_test += lgbm.predict(X_te) / folds.n_splits
        models.append(lgbm)
    # compute OOF SMAPE on original price scale
    oof_price = np.expm1(oof)
    smape_oof = smape(train['price'].values, oof_price)
    print(f"OOF SMAPE = {smape_oof:.4f}%")

    # save artifacts
    os.makedirs('artifacts', exist_ok=True)
    dump(models, 'artifacts/lgb_models.joblib')
    dump(scaler, 'artifacts/scaler.joblib')

    # test predictions
    test_pred_price = np.expm1(preds_test)
    test_pred_price = np.maximum(test_pred_price, 0.01)  # ensure positivity

    # write submission with exact same order/cols as sample_test_out.csv
    out_df = pd.DataFrame({'sample_id': test['sample_id'], 'price': test_pred_price})
    out_path = data_dir / 'test_out.csv'
    out_df.to_csv(out_path, index=False)
    print("Wrote submission to", out_path)

    # save OOF preds for stacking/ensembling
    np.save('artifacts/oof_preds.npy', oof)
    np.save('artifacts/test_preds.npy', preds_test)

    # write a one-page documentation draft
    doc = f"""
# One-page Submission Documentation

**Title:** Smart Product Pricing — Team: [Your Team Name]

**Methodology (short):**
We build a multimodal regression pipeline using only provided dataset. We predict log1p(price) and invert with expm1. Our pipeline uses engineered text features, sentence-transformer embeddings, optional image embeddings (ResNet50), PCA reductions, and LightGBM ensembling via 5-fold OOF.

**Model architecture / algorithms:**
- Base: LightGBM regression models trained on concatenated features (numeric + TF-IDF-SVD + text-emb PCA [+ image-emb PCA]).
- Training: 5-fold stratified CV on binned log(price).
- Postprocessing: inverse transform expm1, enforce min price > 0.01.

**Feature engineering:**
- Parsed Item Pack Quantity (regex)
- Numeric spec flags (GB, mAh, ml, etc.)
- SentenceTransformer text embeddings (reduced to 128 dims)
- TF-IDF reduced via TruncatedSVD (200 dims)
- Optional ResNet50-based image embeddings reduced to 128 dims

**Evaluation:**
- Metric: SMAPE on original price scale.
- OOF SMAPE reported during training: {smape_oof:.4f}%.

**Compliance:**
- No external price lookup or external price datasets used.
- Pretrained encoders used only for representation — final model will be released under MIT/Apache2.0 compatible components only.

**Reproducibility:**
- Artifacts: models in /artifacts, oof/test preds saved.
- Run: python src/train_and_infer.py --mode full --use_text_emb True [--use_images True]

"""
    with open('Documentation_submission.md','w') as f:
        f.write(doc.strip())
    print("Wrote Documentation_submission.md")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['quick','full'], default='quick')
    p.add_argument('--use_text_emb', default=False, action='store_true')
    p.add_argument('--text_emb_model', default='all-MiniLM-L6-v2')
    p.add_argument('--use_images', default=False, action='store_true')
    p.add_argument('--use_cuda', default=False, action='store_true')
    args = p.parse_args()
    main(args)
