"""Tối ưu siêu tham số Random Forest có ràng buộc bằng Optuna TPE, giữ riêng tập test."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RANGES = {'n_estimators': (10, 300), 'max_depth': (2, 25), 'min_samples_split': (2, 20), 'min_samples_leaf': (1, 15), 'max_features': (0.2, 1.0)}

def split_data(seed):
    data = load_breast_cancer()
    X_a, X_test, y_a, y_test = train_test_split(data.data, data.target, test_size=0.15, stratify=data.target, random_state=seed)
    # Train ~70%, validation ~15%, test ~15%.
    X_train, X_val, y_train, y_val = train_test_split(X_a, y_a, test_size=0.15/0.85, stratify=y_a, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

def model_for(p, seed):
    return RandomForestClassifier(**p, random_state=seed, n_jobs=1)

def fixed_batch(X, seed, size=1000):
    rng = np.random.default_rng(seed+777)
    return np.ascontiguousarray(X[rng.choice(len(X), size=size, replace=True)])

def latency_ms(model, batch, repeats):
    model.predict(batch)
    times = []
    for _ in range(repeats):
        tic = time.perf_counter()
        model.predict(batch)
        times.append((time.perf_counter()-tic)*1000)
    return float(np.median(times)), float(np.std(times, ddof=1)) if len(times)>1 else 0.0

def evaluate(p, X_train, y_train, X_val, y_val, batch, seed, repeats, bound):
    model = model_for(p, seed)
    tic = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter()-tic
    f1 = float(f1_score(y_val, model.predict(X_val)))
    latency, latency_sd = latency_ms(model, batch, repeats)
    feasible = latency <= bound
    error = 1.0-f1
    # Ưu tiên nghiệm thỏa ràng buộc; nếu không thỏa, ưu tiên vi phạm nhỏ hơn.
    loss = error if feasible else 1.0 + (latency-bound)/bound + error
    return {'f1_val':f1, 'val_error':error, 'latency_ms':latency, 'latency_std_ms':latency_sd,
            'is_feasible': bool(feasible), 'constraint_violation_ms':max(0.0,latency-bound),
            'search_loss':loss, 'fit_seconds':fit_seconds}

def choose_best(df):
    feasible = df[df.is_feasible]
    # Chọn nghiệm khả thi trước, sau đó ưu tiên F1, độ trễ và số thứ tự thử.
    if len(feasible):
        return feasible.sort_values(['f1_val','latency_ms','trial'], ascending=[False,True,True]).iloc[0]
    return df.sort_values(['constraint_violation_ms','f1_val','trial'],ascending=[True,False,True]).iloc[0]

def params_from(row):
    return {k:(float(row[k]) if k=='max_features' else int(row[k])) for k in RANGES}

def make_plots(log, chosen, bound):
    figdir=ROOT/'figures'; figdir.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8.6,5))
    for method,g in log.groupby('method',sort=False):
        best=np.inf; curve=[]
        for _,r in g.sort_values('trial').iterrows():
            if bool(r['is_feasible']): best=min(best,float(r['val_error']))
            curve.append(best if np.isfinite(best) else np.nan)
        ax.plot(g['trial'],curve,label=method,lw=2)
    ax.set(xlabel='Số lần thử',ylabel='Sai số kiểm định tốt nhất (1 - F1)',title='Quá trình cải thiện nghiệm khả thi')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'convergence_history.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.6,5.2))
    for method,g in log.groupby('method',sort=False):
        ax.scatter(g.latency_ms,g.f1_val,label=method,alpha=.55,s=27)
    f=log[log.is_feasible].sort_values(['latency_ms','f1_val'],ascending=[True,False]); front=[]; top=-np.inf
    for _,r in f.iterrows():
        if r.f1_val>top+1e-12: front.append(r);top=r.f1_val
    if front:
        ax.plot([r.latency_ms for r in front],[r.f1_val for r in front],color='black',lw=1.5,label='Biên không bị trội (thực nghiệm)')
    ax.axvline(bound,ls='--',color='firebrick',label=f'Giới hạn độ trễ: {bound:g} ms')
    ax.scatter([chosen.latency_ms],[chosen.f1_val],marker='*',s=220,color='gold',edgecolors='black',zorder=9,label='Nghiệm TPE được chọn')
    ax.set(xlabel='Trung vị thời gian dự đoán 1.000 mẫu (ms)',ylabel='F1 trên tập kiểm định',title='Đánh đổi giữa F1 và độ trễ suy luận')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figdir/'pareto_accuracy_latency.png',dpi=180);plt.close(fig)

def run(args):
    if args.trials<1 or args.repeats<1 or args.tau_max<=0: raise ValueError('Trials, repeats and tau_max must be positive')
    (ROOT/'results').mkdir(exist_ok=True)
    X_train,X_val,X_test,y_train,y_val,y_test=split_data(args.seed)
    batch=fixed_batch(X_val,args.seed)
    rows=[]; rng=np.random.default_rng(args.seed)
    tic=time.perf_counter()
    for i in range(args.trials):
        p={k:(float(rng.uniform(*v)) if k=='max_features' else int(rng.integers(v[0],v[1]+1))) for k,v in RANGES.items()}
        score=evaluate(p,X_train,y_train,X_val,y_val,batch,args.seed,args.repeats,args.tau_max)
        rows.append(dict(method='Random Search',trial=i+1,**p,**score))
        if (i+1)%10==0:print('Random Search',i+1,flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=args.seed))
    def objective(t):
        p={k:(t.suggest_float(k,*v) if k=='max_features' else t.suggest_int(k,*v)) for k,v in RANGES.items()}
        score=evaluate(p,X_train,y_train,X_val,y_val,batch,args.seed,args.repeats,args.tau_max)
        rows.append(dict(method='Optuna TPE',trial=t.number+1,**p,**score))
        if (t.number+1)%10==0:print('Optuna TPE',t.number+1,flush=True)
        return score['search_loss']
    study.optimize(objective,n_trials=args.trials)
    search_seconds=time.perf_counter()-tic
    log=pd.DataFrame(rows); log.to_csv(ROOT/'results'/'trials_log.csv',index=False)
    selected={m:choose_best(log[log.method==m]) for m in ('Random Search','Optuna TPE')}
    default={'n_estimators':100,'max_depth':None,'min_samples_split':2,'min_samples_leaf':1,'max_features':'sqrt'}
    summary=[]; configs={'Default Baseline':default,**{m:params_from(r) for m,r in selected.items()}}
    # Không dùng test khi tìm kiếm; chỉ đánh giá sau khi chọn cấu hình từ validation.
    for name,p in configs.items():
        model=model_for(p,args.seed); model.fit(X_train,y_train)
        pred=model.predict(X_test)
        # Dùng cùng tập 1.000 hàng từ validation để so sánh độ trễ.
        med,sd=latency_ms(model,batch,max(11,args.repeats))
        summary.append({'method':name,'test_f1':f1_score(y_test,pred),'test_accuracy':accuracy_score(y_test,pred),
                        'test_precision':precision_score(y_test,pred),'test_recall':recall_score(y_test,pred),
                        'latency_ms':med,'latency_std_ms':sd,'latency_feasible':bool(med<=args.tau_max),
                        'selected_trial':int(selected[name].trial) if name in selected else 0,
                        'validation_f1':float(selected[name].f1_val) if name in selected else None})
    summary=pd.DataFrame(summary);summary.to_csv(ROOT/'results'/'test_benchmark_summary.csv',index=False)
    make_plots(log,selected['Optuna TPE'],args.tau_max)
    metadata={'seed':args.seed,'trials_per_method':args.trials,'repeats_per_trial':args.repeats,'latency_budget_ms':args.tau_max,
              'latency_batch_size':1000,'latency_batch_source':'Lấy mẫu có hoàn lại từ tập kiểm định (validation), cố định bằng seed',
              'data_samples':[len(X_train),len(X_val),len(X_test)], 'total_search_seconds':search_seconds,
              'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),
              'sklearn':sklearn.__version__,'optuna':optuna.__version__,
              'selected':{m:{'trial':int(r.trial),'parameters':params_from(r),'validation_f1':float(r.f1_val),
                              'validation_latency_ms':float(r.latency_ms),'validation_feasible':bool(r.is_feasible)} for m,r in selected.items()},
              'notes':['Độ trễ phụ thuộc vào phần cứng và lịch thực thi CPU.',
                       'Bài tập phân loại dữ liệu học thuật, không phải thử nghiệm triển khai chẩn đoán y tế.']}
    (ROOT/'results'/'experiment_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print(summary.to_string(index=False));print('Thời gian tìm kiếm (giây):',round(search_seconds,1))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--trials',type=int,default=50);ap.add_argument('--tau_max',type=float,default=30.)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--repeats',type=int,default=5)
    run(ap.parse_args())
