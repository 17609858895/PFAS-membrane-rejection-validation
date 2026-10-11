"""Reviewer-audit corrections. Writes only inside this isolated revision package.

Primary ranking estimand: equal-study mean of within-study censor-compatible
concordance, conditional on at least one comparable pair. No cross-study pairs.
The 5,000-replicate study bootstrap conditions on the stored out-of-fold fits;
it is not a bootstrap that retrains the complete model-selection procedure.
"""
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import generate_pfas_paper_figures as core
import run_reviewer_requested_checks as checks
import run_major_revision_robustness as major
import audit_internal_high_accuracy_claim as audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'04_Tables'/'Consistency_20260905'
SEED=20260831

def save(frame, name):
    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    print(f'{name}: {len(frame)} rows',flush=True)

def snapshot(raw,df):
    mask=raw['Rejection'].ge(.5)&raw['mem_class'].astype(str).str.upper().isin(['NF','RO'])
    result=df.copy()
    result.insert(0,'source_sheet_row',np.flatnonzero(mask)+2)
    result.insert(0,'analysis_row_id',np.arange(len(df)))
    save(result,'filtered_analysis_snapshot.csv')
    meta={'workbook':'05_Source_Data/es6c02287_si_002.xlsx','sheet':'PFAS_literature_dataset_2b','workbook_sha256':hashlib.sha256(core.WORKBOOK_PATH.read_bytes()).hexdigest(),'analysis_rows':len(df),'studies':int(df.ref_id.nunique()),'analysis_row_id':'Zero-based position after the documented filters; stable for this source hash.','source_sheet_row':'One-based worksheet row, including header row 1.','filter':'Rejection >= 0.50 and membrane class NF/RO; no manual exclusions.'}
    (OUT/'snapshot_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

def table1_sd(df,x):
    scaled=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(x))
    clusters=KMeans(n_clusters=10,random_state=42,n_init=20).fit_predict(scaled)
    masks={}
    membership=[]
    for name,stratify in [('Cluster-stratified',clusters),('Row-random',None)]:
        mask=np.zeros(len(df),bool)
        for rep in range(12):
            _,te=train_test_split(np.arange(len(df)),test_size=.15,random_state=110+rep,stratify=stratify)
            mask[te]=True
            membership.extend({'design':name,'repeat':rep+1,'analysis_row_id':int(i),'partition':'test'} for i in te)
        masks[name]=mask
    for name in ['Study-blocked','Membrane-blocked','PFAS-blocked']:masks[name]=np.ones(len(df),bool)
    masks['Publication-year holdout']=df.ref_year.gt(2023).to_numpy()
    original=pd.read_csv(ROOT/'04_Tables/reviewer_m1_claim_matched_metrics.csv')
    rows=[]
    for _,row in original.iterrows():
        name=row['strategy']; mask=masks[name]
        values=df.loc[mask & ~df.is_censored.to_numpy(),'Rejection']
        assert len(values)==int(row.n_exact),(name,len(values),row.n_exact)
        assert int(mask.sum())==int(row.n),(name,mask.sum(),row.n)
        sd=float(values.std(ddof=1)*100)
        rows.append({**row.to_dict(),'exact_outcome_sd_pp':sd,'nrmse_by_test_sd':float(row.rmse_pp/sd),'aggregation':'one average out-of-fold prediction per tested row' if name in ['Cluster-stratified','Row-random'] else 'one held-out prediction per row'})
    save(pd.DataFrame(rows),'Table_C01_table1_sd_nrmse.csv')
    save(pd.DataFrame(membership),'Table_C02_repeated_test_row_memberships.csv')

def paired_gaps():
    data=pd.read_csv(ROOT/'04_Tables/Revision/Table_R3_05_repeated_50_study_partition_metrics.csv')
    pivot=data.pivot(index='repeat',columns='evaluation_regime',values=['r2','logit_r2','rmse_pp','mae_pp','nrmse_by_test_sd','c_index'])
    differences=pd.DataFrame({m:pivot[(m,'familiar_rows')]-pivot[(m,'unseen_studies')] for m in ['r2','logit_r2','rmse_pp','mae_pp','nrmse_by_test_sd','c_index']}).reset_index()
    save(differences,'Table_C03_paired_partition_differences.csv')
    summary=[]
    for m in differences.columns[1:]:
        s=differences[m]
        summary.append({'metric':m,'direction':'familiar minus unseen','median':s.median(),'q025':s.quantile(.025),'q975':s.quantile(.975),'positive_repeats':int((s>0).sum()),'negative_repeats':int((s<0).sum()),'repeats':len(s)})
    save(pd.DataFrame(summary),'Table_C04_paired_partition_summary.csv')

def ranking_and_bootstrap(df,x):
    summary,predictions=checks.study_blocked_baselines_and_retransformation(df,x)
    groups=df.ref_id.astype(str).to_numpy(); labels=np.unique(groups)
    y=df.Rejection.to_numpy(float); c=df.is_censored.to_numpy(bool)
    fold_id=np.full(len(df),-1,int)
    for fold,(_,te) in enumerate(GroupKFold(n_splits=10).split(x,y,groups),1):fold_id[te]=fold
    save(pd.DataFrame({'analysis_row_id':np.arange(len(df)),'study':groups,'outer_fold':fold_id,'rejection':y,'is_censored':c,**predictions}),'Table_C05_baseline_row_predictions.csv')
    study_rows=[]; macro={}
    for name,pred in predictions.items():
        vals=[]
        for g in labels:
            ix=np.flatnonzero(groups==g)
            numerator=denominator=0.
            for k in range(len(ix)-1):
                a,b=major.pair_concordance(y[ix[k:k+1]],pred[ix[k:k+1]],c[ix[k:k+1]],y[ix[k+1:]],pred[ix[k+1:]],c[ix[k+1:]])
                numerator+=a;denominator+=b
            value=numerator/denominator if denominator else np.nan
            vals.append(value)
            study_rows.append({'model':name,'study':g,'outer_fold':int(fold_id[ix[0]]),'rows':len(ix),'comparable_pairs':int(denominator),'concordance_numerator':numerator,'within_study_c':value})
        macro[name]=np.array(vals)
    save(pd.DataFrame(study_rows),'Table_C06_within_study_concordance.csv')
    for name,s in macro.items():
        sel=summary.model.eq(name)
        summary.loc[sel,'within_study_c_mean']=np.nanmean(s)
        summary.loc[sel,'ranking_eligible_studies']=np.isfinite(s).sum()
    save(summary,'Table_C07_baseline_metrics_consistent.csv')
    rng=np.random.default_rng(SEED+40000)
    draws=rng.integers(0,len(labels),size=(5000,len(labels)))
    counts=np.apply_along_axis(lambda z:np.bincount(z,minlength=len(labels)),1,draws).astype(float)
    save(pd.DataFrame(counts.astype(int),columns=[f'study_{i:02d}' for i in range(len(labels))]).assign(replicate=np.arange(1,5001)),'Table_C08_bootstrap_study_multiplicities.csv')
    save(pd.DataFrame({'column':[f'study_{i:02d}' for i in range(len(labels))],'study':labels}),'Table_C09_bootstrap_study_key.csv')
    group_no=np.searchsorted(labels,groups)
    weights=counts[:,group_no];weights[:,c]=0
    n=weights.sum(axis=1);means=(weights@y)/n
    sst=(weights*(y[None,:]-means[:,None])**2).sum(axis=1)
    names=['13-feature censor-aware AFT','Training-set exact-row mean']
    arrays={}; observed={}
    for name in names:
        error=predictions[name]-y;sse=weights@(error**2)
        valid=np.isfinite(macro[name])
        arrays[name]={'R2':1-sse/sst,'RMSE (pp)':np.sqrt(sse/n)*100,'MAE (pp)':(weights@abs(error))/n*100,'Within-study C-index':(counts[:,valid]@macro[name][valid])/counts[:,valid].sum(axis=1)}
        row=summary.loc[summary.model.eq(name)].iloc[0]
        observed[name]={'R2':row.r2,'RMSE (pp)':row.rmse_pp,'MAE (pp)':row.mae_pp,'Within-study C-index':float(np.nanmean(macro[name]))}
    rows=[];delta={}
    for m in arrays[names[0]]:
        d=arrays[names[0]][m]-arrays[names[1]][m];delta[m]=d
        improving=d<0 if m in ['RMSE (pp)','MAE (pp)'] else d>0
        rows.append({'contrast':m,'observed_difference':observed[names[0]][m]-observed[names[1]][m],'q025':np.quantile(d,.025),'median':np.median(d),'q975':np.quantile(d,.975),'fraction_improves':improving.mean(),'replicates':5000})
    save(pd.DataFrame(rows),'Table_C10_paired_bootstrap_corrected.csv')
    save(pd.DataFrame(delta).assign(replicate=np.arange(1,5001)),'Table_C11_paired_bootstrap_draws.csv')
    (OUT/'ranking_estimand.json').write_text(json.dumps({'primary_ranking_estimand':'Equal-weight mean of within-study C-index across studies with >=1 comparable pair; no cross-study pairs.','eligible_studies':int(np.isfinite(macro[names[0]]).sum()),'all_studies':len(labels),'bootstrap':'5000 draws of 61 studies with replacement; frozen out-of-fold predictions; exclude unscorable studies only from the concordance denominator; row-pooled exact-outcome error metrics.','baseline_within_study_c':float(np.nanmean(macro[names[1]])),'limitation':'Conditional uncertainty for the archived fitting procedure, not full retraining or future-population validation.'},indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

def intervals(df,x):
    """Reproduce the existing study-unit analysis and export every scoring input."""
    groups=df.ref_id.astype(str).to_numpy();y=df.Rejection.to_numpy(float);c=df.is_censored.to_numpy(bool)
    rng=np.random.default_rng(SEED+60000);scores_out=[];members=[];rows_out=[];folds=[]
    for fold,(outer_train,test) in enumerate(GroupKFold(n_splits=10).split(x,y,groups),1):
        shuffled=np.unique(groups[outer_train]).copy();rng.shuffle(shuffled)
        n_cal=min(max(20,int(math.ceil(len(shuffled)*.40))),len(shuffled)-3)
        cal_groups=set(shuffled[:n_cal]);cal=outer_train[np.isin(groups[outer_train],list(cal_groups))];fit=outer_train[~np.isin(groups[outer_train],list(cal_groups))]
        unique_fit=np.unique(groups[fit]); cp=[];tp=[]
        for b in range(12):
            sampled=rng.choice(unique_fit,size=len(unique_fit),replace=True)
            bi=np.concatenate([fit[groups[fit]==g] for g in sampled])
            model=audit.fit('aft',x,df,bi,major.CONSERVATIVE,SEED+61000+fold*100+b)
            cp.append(audit.predict('aft',model,x[cal]));tp.append(audit.predict('aft',model,x[test]))
        cp=np.asarray(cp);tp=np.asarray(tp)
        cm=cp.mean(axis=0);tm=tp.mean(axis=0);cs=np.maximum(cp.std(axis=0,ddof=1),.008);ts=np.maximum(tp.std(axis=0,ddof=1),.008)
        scores=[]
        for g in sorted(cal_groups):
            ix=np.flatnonzero((groups[cal]==g)&~c[cal]);score=float(np.max(abs(y[cal][ix]-cm[ix])/cs[ix])) if len(ix) else np.nan
            scores_out.append({'outer_fold':fold,'study':g,'exact_rows':len(ix),'max_standardized_residual':score})
            if np.isfinite(score):scores.append(score)
        for name,idx in [('fit',fit),('calibration',cal),('test',test)]:
            members.extend({'outer_fold':fold,'study':g,'partition':name} for g in np.unique(groups[idx]))
        for nominal in [.8,.9,.95]:
            q,rank=major.finite_group_quantile(np.array(scores),nominal)
            lo=np.clip(tm-q*ts,0,1);hi=np.clip(tm+q*ts,0,1)
            lo_high=np.clip(tm-q*ts,.5,1);hi_high=np.clip(tm+q*ts,.5,1)
            exact=~c[test];covered=(y[test]>=lo)&(y[test]<=hi)
            study_cover=[bool(covered[(groups[test]==g)&exact].all()) for g in np.unique(groups[test]) if ((groups[test]==g)&exact).any()]
            folds.append({'fold':fold,'nominal':nominal,'calibration_scores':len(scores),'rank':rank,'q':q,'row_coverage':float(covered[exact].mean()),'study_coverage':float(np.mean(study_cover)),'covered_studies':int(np.sum(study_cover)),'eligible_test_studies':len(study_cover),'width_0_1_pp':float(((hi-lo)*100).mean()),'width_05_1_pp':float(((hi_high-lo_high)*100).mean())})
            rows_out.extend({'analysis_row_id':int(i),'fold':fold,'nominal':nominal,'study':groups[i],'rejection':y[i],'is_censored':bool(c[i]),'prediction_mean':float(tm[j]),'local_scale':float(ts[j]),'lo_0_1':float(lo[j]),'hi_0_1':float(hi[j]),'lo_05_1':float(lo_high[j]),'hi_05_1':float(hi_high[j])} for j,i in enumerate(test))
        print(f'interval inputs exported fold {fold}/10',flush=True)
    save(pd.DataFrame(scores_out),'Table_C12_study_calibration_scores.csv')
    save(pd.DataFrame(members),'Table_C13_interval_study_memberships.csv')
    save(pd.DataFrame(rows_out),'Table_C14_interval_row_outputs.csv')
    frame=pd.DataFrame(folds);save(frame,'Table_C15_interval_fold_summary.csv')
    prior=pd.read_csv(ROOT/'04_Tables/Revision/Table_R3_12_study_unit_conformal_folds.csv')
    assert np.allclose(frame.width_0_1_pp,prior.mean_interval_width_pp,atol=1e-5)
    assert np.allclose(frame.study_coverage,prior.study_all_exact_rows_coverage)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw,df=core.load_data();x=core.feature_matrix(df)
    snapshot(raw,df);table1_sd(df,x);paired_gaps();ranking_and_bootstrap(df,x);intervals(df,x)
    print('CONSISTENCY CORRECTIONS COMPLETE',flush=True)

if __name__=='__main__':main()
