import numpy as np, sys, json
sys.path.insert(0,'ml-experiments/00660-shared-generator-of-injected-load-transfer-events')
import e00660_data as d
from studies.sources import ML_EXPERIMENTS_DATA_DIR as D
panel=d.load_panel(); info=d.series_info()
new=D/'00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe/forecast_loyo'
S=2*24*365*2  # 2022-23 = first 2 years in half-hours
def sig(r,lo,hi):
    x=r[lo:hi]; n=len(x)//12*12
    b=np.nanmean(x[:n].reshape(-1,12),axis=1); dd=np.diff(b); dd=dd[np.isfinite(dd)]
    return 1.4826*np.median(np.abs(dd-np.median(dd))), np.nanmean(np.abs(x))
SF=d.idx_of(when=__import__('datetime').datetime(2024,1,1,tzinfo=__import__('datetime').UTC))
print('site  C_2024+ new_2024+  C_22-23 new_22-23   MAE C new (2024+)')
tot=[0,0]
for s,i in info.items():
    p=panel.power[s]; c=d.load_forecast(site=s); n=np.load(new/f'forecast_{i["id"]}.npy')
    rc,rn=p-c,p-n
    a=sig(rc,SF,len(p)); b=sig(rn,SF,len(p)); a2=sig(rc,0,SF); b2=sig(rn,0,SF)
    tot[0]+=b[0]<a[0]; tot[1]+=1
    print(f'{s:5s} {a[0]:.3f} {b[0]:.3f}   {a2[0]:.3f} {b2[0]:.3f}   {a[1]:.3f} {b[1]:.3f}')
print('new lower noise at',tot)
