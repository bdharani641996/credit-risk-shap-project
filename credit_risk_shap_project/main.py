
\"\"\"Credit Risk Modeling + SHAP Analysis - runnable pipeline

How to use:
  1. Create a Python environment and install dependencies:
     pip install -r requirements.txt
  2. Run:
     python main.py --data credit_data.csv --out_dir output
Outputs produced in output/:
  - model.pkl (trained LightGBM cross-validated model or final model)
  - cv_metrics.json (AUC, F1-score on CV)
  - shap_summary.png (global feature importance from SHAP)
  - shap_local_instances.json (textual explanations for three selected instances)
  - selected_instances.csv (three chosen loan application rows: high-risk, low-risk, borderline)
  - report_generated.md (concise report)
\"\"\"

import argparse, os, json
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='credit_data.csv')
    parser.add_argument('--out_dir', default='output')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    try:
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
        import shap
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
    except Exception as e:
        print('Missing required packages. Please install requirements.txt and re-run.')
        print(str(e))
        return

    df = pd.read_csv(args.data)
    target = 'default'
    X = df.drop(columns=[target])
    y = df[target]

    # Simple preprocessing
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    num_transform = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_transform = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preproc = ColumnTransformer([('num', num_transform, num_cols),
                                 ('cat', cat_transform, cat_cols)])

    X_proc = preproc.fit_transform(X)

    # LightGBM dataset and CV training
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'verbosity': -1,
        'seed': 42,
        'n_jobs': -1
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y)):
        X_train, X_val = X_proc[train_idx], X_proc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        model = lgb.train(lgb_params, lgb_train, num_boost_round=1000,
                          valid_sets=[lgb_train,lgb_val],
                          early_stopping_rounds=50, verbose_eval=False)
        y_prob = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred)
        aucs.append(auc); f1s.append(f1)
        print(f'Fold {fold} AUC={auc:.4f} F1={f1:.4f}')

    metrics = {'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
               'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s))}

    with open(os.path.join(out_dir, 'cv_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Train final model on full data
    final_data = lgb.Dataset(X_proc, label=y)
    final_model = lgb.train(lgb_params, final_data, num_boost_round= int(model.best_iteration*1.1))

    # SHAP analysis (global + local)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_proc)[1] if isinstance(explainer.shap_values(X_proc), list) else explainer.shap_values(X_proc)

    # Save model and preprocessor
    import joblib
    joblib.dump(final_model, os.path.join(out_dir, 'model.pkl'))
    joblib.dump(preproc, os.path.join(out_dir, 'preprocessor.pkl'))

    # Global importance: mean(|SHAP|)
    mean_abs = np.abs(shap_values).mean(axis=0)
    # get feature names after preprocessing (onehot features)
    try:
        # OneHotEncoder feature names (scikit-learn >= 1.0)
        ohe = preproc.named_transformers_['cat'].named_steps['onehot']
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    except Exception:
        cat_names = cat_cols
    feature_names = num_cols + cat_names
    # save summary to csv
    import pandas as pd
    gfi = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)
    gfi.to_csv(os.path.join(out_dir, 'shap_global_importance.csv'), index=False)

    # Generate and save summary plot (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        topn = gfi.head(20)
        plt.barh(topn['feature'][::-1], topn['mean_abs_shap'][::-1])
        plt.xlabel('mean(|SHAP value|)')
        plt.title('Global feature importance (SHAP)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'shap_summary.png'), dpi=150)
        plt.close()
    except Exception as e:
        print('Could not create plot. matplotlib required.', str(e))

    # Select three instances: high-risk, low-risk, borderline
    probs = final_model.predict(X_proc)
    df_indices = pd.DataFrame({'prob': probs, 'true': y})
    high_idx = df_indices['prob'].idxmax()
    low_idx = df_indices['prob'].idxmin()
    # borderline: closest to 0.5
    borderline_idx = (np.abs(df_indices['prob'] - 0.5)).idxmin()
    selected = X.iloc[[high_idx, low_idx, borderline_idx]].copy()
    selected['pred_prob'] = probs[[high_idx, low_idx, borderline_idx]]
    selected['true'] = y.iloc[[high_idx, low_idx, borderline_idx]].values
    selected.to_csv(os.path.join(out_dir, 'selected_instances.csv'), index=False)

    # Local explanations: convert shap values for selected to readable form
    local_explanations = {}
    for idx in [high_idx, low_idx, borderline_idx]:
        vals = shap_values[idx]
        # pair feature name with shap value and original value
        pairs = sorted(zip(feature_names, vals, X.iloc[idx].tolist()), key=lambda x: -abs(x[1]))
        local_explanations[int(idx)] = [{'feature': f, 'shap': float(s), 'value': float(v) if isinstance(v,(int,float)) else str(v)} for f,s,v in pairs[:10]]

    with open(os.path.join(out_dir, 'shap_local_instances.json'), 'w') as f:
        json.dump(local_explanations, f, indent=2)

    # Create a short report file summarizing steps and results (automatically generated)
    report = f\"\"\"# Project: Interpretable ML for Credit Risk (SHAP)

    ## What I did (automated)
    - Preprocessing: median-imputed numeric features, most-frequent-imputed then one-hot encoded categorical features.
    - Model: LightGBM (binary classification), 5-fold stratified cross-validation with early stopping.
    - Metrics: AUC and F1 reported from CV, saved to cv_metrics.json.
    - SHAP: TreeExplainer on final model; global importance saved to shap_global_importance.csv; local explanations for three instances saved to shap_local_instances.json.
    - Selected instances (CSV): high-risk, low-risk, borderline saved to selected_instances.csv.

    ## How to reproduce
    1. Install requirements: pip install -r requirements.txt
    2. Run: python main.py --data credit_data.csv --out_dir output

    ## Suggested interpretation steps (manual)
    1. Open 'shap_global_importance.csv' to see top features by mean(|SHAP|).
    2. Inspect 'shap_local_instances.json' for local explanations and reasoning behind each selected prediction.
    3. Use shap.dependence_plot on pairs of features (e.g. revol_util vs dti) to explore interactions.

    ## Notes
    - This pipeline is intentionally explicit and minimal so it runs in typical environments.
    - If you have a real dataset (e.g. LendingClub), replace credit_data.csv with your file and ensure consistent column names.
    \"\"\"

    with open(os.path.join(out_dir, 'report_generated.md'), 'w') as f:
        f.write(report)

    print('Pipeline complete. Outputs in', out_dir)

if __name__ == '__main__':
    main()
