"""Новый baseline: стандартные параметры библиотек, прежние внешние разбиения.
Не перезаписывает результаты основного исследования и не меняет статью.
Запуск: analytics_learning/bin/python 04_baseline_default_run.py
"""
from pathlib import Path
import ast, json, hashlib, platform, warnings, time, os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__import__('tempfile').gettempdir())/'oks-default-baseline-mpl'))
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,confusion_matrix,roc_curve
from sklearn.exceptions import ConvergenceWarning
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
import sklearn,catboost,xgboost
threadpool_limits(2)
from repository_support import initialize
P=initialize('acs')
PREFIX='04_baseline_default_'
def savejson(obj,name):
 (P/(PREFIX+name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(t):print(time.strftime('%H:%M:%S')+' '+t,flush=True)
D=pd.read_csv(P/'01_данные.csv');meta=json.loads((P/'01_конфигурация.json').read_text());features=meta['features'];X=D[features];y=D.hospital_death.to_numpy();groups=D.group_id.to_numpy()
assert sha(P/'01_данные.csv')==meta['data_sha256'];assert np.array_equal(D.row_id,np.arange(len(D)))
outer=pd.read_csv(P/'02_внешние_разбиения.csv');inner=pd.read_csv(P/'02_внутренние_разбиения.csv')
# Reuse only the preprocessing definition; no notebook cells or training searches execute.
nb=json.loads((P/'02_модели_Optuna_и_эксперименты.ipynb').read_text());code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code');tree=ast.parse(code);defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_preprocessor'];assert len(defs)==1
exec(compile(ast.Module(body=defs,type_ignores=[]),str(P/'02_модели_Optuna_и_эксперименты.ipynb'),'exec'))
FAMILIES=['LR','RF','CatBoost','XGBoost'];PARAMS=[];WARNS=[];FITS=[];ROWS=[];METRICS=[]
def new_model(family,seed):
 if family=='LR':return LogisticRegression(random_state=seed)
 if family=='RF':return RandomForestClassifier(random_state=seed,n_jobs=2)
 if family=='CatBoost':return CatBoostClassifier(random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
 return XGBClassifier(random_state=seed,n_jobs=2)
def fit_predict(tr,te,family,seed,stage,r,f,k=None):
 assert not set(groups[tr])&set(groups[te]);pre=make_preprocessor(features,family,True);a=pre.fit_transform(X.iloc[tr]);b=pre.transform(X.iloc[te]);model=new_model(family,seed)
 with warnings.catch_warnings(record=True) as captured:
  warnings.simplefilter('always');model.fit(a,y[tr])
 for w in captured:WARNS.append(dict(family=family,repeat=r,fold=f,inner_fold=k,stage=stage,category=w.category.__name__,message=str(w.message)))
 pred=model.predict_proba(b)[:,1];assert list(model.classes_)==[0,1];assert np.isfinite(pred).all() and np.all((pred>=0)&(pred<=1))
 FITS.append(dict(family=family,repeat=r,fold=f,inner_fold=k,stage=stage,seed=seed,train_n=len(tr),test_n=len(te),train_deaths=int(y[tr].sum()),test_deaths=int(y[te].sum())))
 if stage=='outer':
  entry=dict(family=family,repeat=r,fold=f,seed=seed,estimator_params=model.get_params())
  if family=='CatBoost':entry['resolved_params']=model.get_all_params()
  if family=='XGBoost':entry['resolved_booster_config']=json.loads(model.get_booster().save_config());entry['rounds']=model.get_booster().num_boosted_rounds()
  if family=='LR':entry['n_iter']=model.n_iter_.tolist();entry['reached_max_iter']=bool(np.max(model.n_iter_)>=model.max_iter)
  PARAMS.append(entry)
 return pred

def choose_threshold(target,pred):
 fpr,tpr,t=roc_curve(target,pred,drop_intermediate=False);ba=(tpr+1-fpr)/2;valid=np.isfinite(t);ids=np.flatnonzero(valid & np.isclose(ba,ba[valid].max(),atol=1e-12,rtol=0));return float(t[ids[np.argmax(tpr[ids])]])
def metrics(target,pred,threshold):
 tn,fp,fn,tp=confusion_matrix(target,pred>=threshold,labels=[0,1]).ravel()
 return dict(AP=average_precision_score(target,pred),ROC_AUC=roc_auc_score(target,pred),Brier=brier_score_loss(target,pred),recall=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/(tp+fp) if tp+fp else 0,TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),n=len(target),deaths=int(sum(target)),mean_probability=float(np.mean(pred)))
protocol=dict(experiment='library_defaults_oks_2026_10_28',python=platform.python_version(),versions=dict(sklearn=sklearn.__version__,catboost=catboost.__version__,xgboost=xgboost.__version__,numpy=np.__version__,pandas=pd.__version__),data_sha256=sha(P/'01_данные.csv'),outer_splits_sha256=sha(P/'02_внешние_разбиения.csv'),inner_splits_sha256=sha(P/'02_внутренние_разбиения.csv'),script_sha256=sha(__file__),features=features,preprocessing='original OКС preprocessing; all 15 missing flags; imputation training only; scaling numeric for LR only',training='all actual training rows; no class weights or sampling',threshold='max balanced accuracy inner OOF; tie larger recall',comparison='saved tuned LR RF CatBoost XGBoost only; former fixed baseline excluded',primary='arithmetic mean metrics of 9 outer folds',secondary='mean metrics of 3 full pooled OOF repeats; not equivalent for AP/AUC',default_overrides='only RNG seed, CPU parallelism and logging/file output; no statistical model hyperparameter overrides',initial_estimator_params={f:new_model(f,2026).get_params() for f in FAMILIES},status='protocol fixed before evaluation')
savejson(protocol,'protocol');log('Protocol saved before new evaluation')
for (r,f),part in outer.groupby(['repeat','fold'],sort=True):
 tr=part.loc[part.side=='train','row_id'].to_numpy();te=part.loc[part.side=='test','row_id'].to_numpy();seed=2026+100*int(r)+10*int(f);ins=inner[(inner.repeat==r)&(inner.fold==f)]
 for family in FAMILIES:
  ip=np.full(len(D),np.nan)
  for k,v in ins.groupby('inner_fold',sort=True):
   it=v.loc[v.side=='train','row_id'].to_numpy();iv=v.loc[v.side=='validation','row_id'].to_numpy();assert set(it)|set(iv)==set(tr)
   ip[iv]=fit_predict(it,iv,family,seed+int(k),'inner',int(r),int(f),int(k))
  assert np.isfinite(ip[tr]).all() and np.isnan(ip[te]).all();threshold=choose_threshold(y[tr],ip[tr]);pred=fit_predict(tr,te,family,seed,'outer',int(r),int(f))
  ROWS.extend(dict(row_id=int(j),group_id=int(groups[j]),repeat=int(r),fold=int(f),family=family,y_true=int(y[j]),p_death=float(p),threshold=threshold) for j,p in zip(te,pred))
  m=metrics(y[te],pred,threshold);METRICS.append(dict(repeat=int(r),fold=int(f),family=family,threshold=threshold,**m));log(f'r{r} f{f} {family}: AP={m["AP"]:.5f}; ROC-AUC={m["ROC_AUC"]:.5f}; recall={m["recall"]:.5f}')
  pd.DataFrame(ROWS).to_csv(P/(PREFIX+'oof.csv'),index=False);pd.DataFrame(METRICS).to_csv(P/(PREFIX+'fold_metrics.csv'),index=False);savejson(PARAMS,'resolved_parameters');savejson(WARNS,'warnings');savejson(FITS,'fit_audit')
O=pd.DataFrame(ROWS);assert len(O)==len(D)*3*4;assert not O.duplicated(['family','repeat','row_id']).any();assert O.groupby(['family','repeat']).row_id.nunique().eq(len(D)).all()
SAVED=pd.read_csv(P/'02_oof_predictions.csv');T=SAVED[SAVED.family.isin(FAMILIES)].copy();assert len(T)==len(O)
merged=O.merge(T,on=['row_id','repeat','fold','family'],suffixes=('_default','_tuned'),validate='one_to_one');assert np.array_equal(merged.y_true_default,merged.y_true_tuned)
ALL=[]
for label,frame in [('default',O),('tuned',T)]:
 for (family,r,f),v in frame.groupby(['family','repeat','fold']):ALL.append(dict(variant=label,family=family,repeat=r,fold=f,**metrics(v.y_true.to_numpy(),v.p_death.to_numpy(),v.threshold.to_numpy())))
A=pd.DataFrame(ALL);A.to_csv(P/(PREFIX+'comparison_folds.csv'),index=False);cols=['AP','ROC_AUC','Brier','recall','specificity','precision'];summary=A.groupby(['family','variant'])[cols].mean();summary.to_csv(P/(PREFIX+'comparison_summary.csv'))
pools=[]
for label,frame in [('default',O),('tuned',T)]:
 for (family,r),v in frame.groupby(['family','repeat']):pools.append(dict(variant=label,family=family,repeat=r,**metrics(v.y_true.to_numpy(),v.p_death.to_numpy(),v.threshold.to_numpy())))
PP=pd.DataFrame(pools);PP.to_csv(P/(PREFIX+'pooled_repeats.csv'),index=False);PP.groupby(['family','variant'])[cols].mean().to_csv(P/(PREFIX+'pooled_summary.csv'))
paired=[]
for family in FAMILIES:
 for metric in cols:
  v=A[A.family==family].pivot(index=['repeat','fold'],columns='variant',values=metric);diff=v.tuned-v.default;paired.append(dict(family=family,metric=metric,mean_default=v.default.mean(),mean_tuned=v.tuned.mean(),delta_tuned_minus_default=diff.mean(),folds_tuned_higher=int((diff>1e-12).sum()),folds_equal=int((abs(diff)<=1e-12).sum()),folds_tuned_lower=int((diff < -1e-12).sum())))
pd.DataFrame(paired).to_csv(P/(PREFIX+'paired_differences.csv'),index=False)
protocol.update(status='completed',fits=len(FITS),predictions=len(O),warning_count=len(WARNS),outputs_sha256={f:sha(P/(PREFIX+f)) for f in ['oof.csv','fold_metrics.csv','comparison_summary.csv','paired_differences.csv']});savejson(protocol,'protocol');log('COMPLETE: 144 fits; no prior fixed baseline comparison; no article modification');print(summary.round(6).to_string(),flush=True)
