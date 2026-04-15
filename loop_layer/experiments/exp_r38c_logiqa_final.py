"""
R38c: 补全 LogiQA 评测 + 生成最终汇总可视化
================================================================
1. 对 LogiQA 运行完整 R38a 条件（7 conditions）
2. 将结果合并进 r38_signal_full_bench_results.json
3. 生成最终 5 种可视化（在 R38a 基础上含 LogiQA）

R38b 的经验教训（已并入分析）：
  min_start 必须 ≥ 9，排除早期层（L6-L8）高 cos_res 假阳性峰。
  扩展 n_t 到 14 会被早期层假阳性占据，性能崩溃。
  → R38a 的 min_start=9, n_t∈{4,6,8} 是正确的约束。
"""
from __future__ import annotations

import json, os, sys, time, warnings, re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp/loop_layer")
EXP  = ROOT / "experiments"
ETD  = ROOT / "ETD"
for p in (str(ROOT), str(EXP), str(ETD)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from etd_forward import etd_forward_logits, baseline_forward_logits

MODEL_PATH  = "/root/autodl-tmp/model_qwen"
RESULTS_DIR = EXP / "results"
FIGURES_DIR = EXP / "figures" / "r38_signal_full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE   = torch.bfloat16
N_CALIB = 20
K_ETD   = 2
MIN_START = 9
MAX_START = 22
PROBE_LAYERS = list(range(6, 29, 2))

SWEEP_BEST = {
    "BoolQ": (8,22), "ARC-C": (14,20), "TruthfulQA": (16,19),
    "CSQA": (10,22), "MMLU-HS-Math": (10,18), "GPQA-Diamond": (18,20),
    "AGIEval-Gaokao-MathQA": (13,20), "LogiQA": (14,19),
}

COND_NAMES = ["baseline","sweep_best","persample_cos8","persample_var",
              "onset_fixed8","calib_onset8","calib_global8"]
COND_COLORS = {
    "baseline":"#9E9E9E","sweep_best":"#2196F3","persample_cos8":"#4CAF50",
    "persample_var":"#8BC34A","onset_fixed8":"#FF9800","calib_onset8":"#F44336",
    "calib_global8":"#9C27B0",
}
COND_LABELS = {
    "baseline":"Baseline","sweep_best":"扫参最优","persample_cos8":"逐样本-8层",
    "persample_var":"逐样本-变长","onset_fixed8":"固定Onset-8",
    "calib_onset8":"标定Onset-8","calib_global8":"标定全局-8",
}
BENCH_COLORS = {
    "BoolQ":"#2196F3","ARC-C":"#F44336","CSQA":"#4CAF50","TruthfulQA":"#FF9800",
    "MMLU-HS-Math":"#9C27B0","GPQA-Diamond":"#00BCD4",
    "AGIEval-Gaokao-MathQA":"#795548","LogiQA":"#607D8B",
}

# ─── LogiQA 加载（直接用 fireworks-ai/logiqa）──────────────────────────────────
def load_logiqa_fixed(n):
    def _strip(opt): return re.sub(r"^[ABCDabcd]\.\s*","",str(opt).strip())
    def _to_letter(r):
        lab = r.get("label") if r.get("label") is not None else r.get("answer")
        if isinstance(lab,(int,float)) and lab==int(lab):
            i=int(lab)
            if 0<=i<4: return "abcd"[i]
        s=str(lab).strip().lower()
        if s in "abcd": return s
        return None
    ds = load_dataset("fireworks-ai/logiqa", split="test")
    out = []
    for r in ds:
        label=_to_letter(r)
        if label is None: continue
        opts=r["options"]
        if hasattr(opts,"tolist"): opts=opts.tolist()
        choices=["a","b","c","d"]
        prompt=(f"Passage: {r['context']}\nQuestion: {r['question']}\nChoices:\n"
                +"\n".join(f"{l.upper()}. {_strip(o)}" for l,o in zip(choices,opts))
                +"\nAnswer:")
        out.append({"prompt":prompt,"choices":choices,"label":choices.index(label)})
        if len(out)>=n: break
    return out


# ─── 工具函数 ──────────────────────────────────────────────────────────────────
def safe_cos(u,v):
    u=u.float().reshape(-1).cpu(); v=v.float().reshape(-1).cpu()
    nu,nv=u.norm(),v.norm()
    if nu<1e-12 or nv<1e-12: return 0.0
    return float((u@v/(nu*nv)).clamp(-1,1).item())

def loglikelihood_mc(logits,input_ids,prompt_len):
    total=0.0
    for i in range(prompt_len,input_ids.shape[1]):
        logp=F.log_softmax(logits[0,i-1].float(),dim=-1)
        total+=float(logp[input_ids[0,i]].item())
    return total

def load_model():
    print("Loading Qwen3-8B …")
    tok=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)
    mdl=AutoModelForCausalLM.from_pretrained(MODEL_PATH,torch_dtype=DTYPE,device_map="auto",
                                              attn_implementation="eager",trust_remote_code=True)
    mdl.eval()
    n_layers=mdl.config.num_hidden_layers
    print(f"  {n_layers} layers, device={DEVICE}")
    return tok,mdl,n_layers

@torch.no_grad()
def probe_forward_collect_cos_res(model,input_ids,attn_mask,n_layers):
    base=model.model
    h_inputs,a_outputs,m_outputs={},{},{}
    hooks=[]
    for li in range(n_layers):
        layer=base.layers[li]
        def make_pre(idx):
            def fn(_m,args):
                t=args[0] if isinstance(args,tuple) else args
                h_inputs[idx]=t[:,-1:,:].detach().clone()
            return fn
        def make_attn_post(idx):
            def fn(_m,_i,out):
                t=out[0] if isinstance(out,tuple) else out
                a_outputs[idx]=t[:,-1:,:].detach().clone()
            return fn
        def make_mlp_post(idx):
            def fn(_m,_i,out):
                m_outputs[idx]=out[:,-1:,:].detach().clone()
            return fn
        hooks+=[layer.register_forward_pre_hook(make_pre(li)),
                layer.self_attn.register_forward_hook(make_attn_post(li)),
                layer.mlp.register_forward_hook(make_mlp_post(li))]
    model(input_ids=input_ids,attention_mask=attn_mask,use_cache=False)
    for h in hooks: h.remove()
    cos_res={}
    for li in PROBE_LAYERS:
        hi,al,ml=h_inputs.get(li),a_outputs.get(li),m_outputs.get(li)
        if hi is None or al is None or ml is None: continue
        try:
            layer=base.layers[li]
            m_l0=layer.mlp(layer.post_attention_layernorm(hi))
            term1=(ml-m_l0).squeeze()
            delta_h=(al+ml).squeeze()
            cos_res[li]=safe_cos(term1,delta_h)
        except Exception: pass
    return cos_res

def calibrate_benchmark_profile(items,model,tok,n_layers):
    acc=defaultdict(list)
    for item in items[:N_CALIB]:
        enc=tok(item["prompt"],return_tensors="pt",add_special_tokens=False)
        ids=enc["input_ids"].to(DEVICE)
        amask=enc.get("attention_mask")
        if amask is not None: amask=amask.to(DEVICE)
        cr=probe_forward_collect_cos_res(model,ids,amask,n_layers)
        for li,v in cr.items(): acc[li].append(v)
    return {li:float(np.mean(vs)) for li,vs in acc.items() if vs}

def derive_global_window(profile,n_t=8,min_start=MIN_START,max_start=MAX_START):
    best_start=min_start; best_score=-999.0
    for start in range(min_start,max_start+1):
        vals=[profile[l] for l in profile if start<=l<start+n_t]
        if len(vals)<2: continue
        score=float(np.mean(vals))
        if score>best_score: best_score=score; best_start=start
    return best_start,best_start+n_t

def derive_onset_adaptive(profile,ratio=0.65,n_t=8,min_start=MIN_START,max_start=MAX_START):
    valid={l:v for l,v in profile.items() if min_start<=l<=max_start}
    if not valid: return min_start,min_start+n_t
    max_val=max(valid.values()); thr=max_val*ratio
    for l in sorted(valid):
        if valid[l]>=thr: return l,l+n_t
    t=max(valid,key=valid.__getitem__)
    return t,t+n_t

def select_window_persample(cos_res,n_t=8,min_start=MIN_START,max_start=MAX_START):
    best_start=min_start; best_score=-999.0
    for start in range(min_start,max_start+1):
        vals=[cos_res[l] for l in cos_res if start<=l<start+n_t]
        if len(vals)<2: continue
        score=float(np.mean(vals))
        if score>best_score: best_score=score; best_start=start
    return best_start,best_start+n_t

def select_window_variable_nt(cos_res,nt_candidates=(4,6,8),min_start=MIN_START,max_start=MAX_START):
    best_start=min_start; best_nt=nt_candidates[0]; best_score=-999.0
    for n_t in nt_candidates:
        for start in range(min_start,max_start+1):
            vals=[cos_res[l] for l in cos_res if start<=l<start+n_t]
            if len(vals)<2: continue
            score=float(np.mean(vals))
            if score>best_score: best_score=score; best_start=start; best_nt=n_t
    return best_start,best_start+best_nt

def select_window_onset_fixed(cos_res,threshold=0.28,n_t=8,min_start=MIN_START,max_start=MAX_START):
    for l in sorted(l for l in cos_res if min_start<=l<=max_start):
        if cos_res[l]>=threshold: return l,l+n_t
    return select_window_persample(cos_res,n_t,min_start,max_start)

def mc_predict(model,tok,item,n_e=None,n_t=None,k=K_ETD):
    prompt,choices=item["prompt"],item["choices"]
    scores=[]
    for cont in choices:
        full=prompt+" "+cont
        enc=tok(full,return_tensors="pt",add_special_tokens=False)
        ids=enc["input_ids"].to(DEVICE)
        amask=enc.get("attention_mask")
        if amask is not None: amask=amask.to(DEVICE)
        plen=tok(prompt,return_tensors="pt",add_special_tokens=False)["input_ids"].shape[1]
        if n_e is not None and n_t is not None and n_t>0:
            alpha=min(1.0,6.0/max(n_t,1))
            lgts=etd_forward_logits(model,ids,amask,n_e=n_e,n_t=n_t,k=k,alpha=alpha)
        else:
            lgts=baseline_forward_logits(model,ids,amask)
        scores.append(loglikelihood_mc(lgts,ids,plen))
    return int(np.argmax(scores))

def evaluate_logiqa(items,model,tok,n_layers):
    n_total=len(items)
    sweep_win=SWEEP_BEST["LogiQA"]

    print("  [标定] 聚合前 20 样本 cos_res profile …")
    t0c=time.time()
    mean_profile=calibrate_benchmark_profile(items,model,tok,n_layers)
    calib_global8_win=derive_global_window(mean_profile,n_t=8)
    calib_onset8_win=derive_onset_adaptive(mean_profile,ratio=0.65,n_t=8)
    print(f"  [标定] 完成 ({time.time()-t0c:.1f}s)")
    print(f"    calib_global8={calib_global8_win}  calib_onset8={calib_onset8_win}  sweep={sweep_win}")

    correct={c:0 for c in COND_NAMES}
    selected_tstarts={c:[] for c in ["persample_cos8","persample_var","onset_fixed8"]}
    t0=time.time()
    for i,item in enumerate(items):
        label=item["label"]
        enc=tok(item["prompt"],return_tensors="pt",add_special_tokens=False)
        probe_ids=enc["input_ids"].to(DEVICE)
        probe_mask=enc.get("attention_mask")
        if probe_mask is not None: probe_mask=probe_mask.to(DEVICE)
        cos_res=probe_forward_collect_cos_res(model,probe_ids,probe_mask,n_layers)

        ps8_win=select_window_persample(cos_res,n_t=8)
        var_win=select_window_variable_nt(cos_res,nt_candidates=(4,6,8))
        on8_win=select_window_onset_fixed(cos_res,threshold=0.28,n_t=8)
        selected_tstarts["persample_cos8"].append(ps8_win[0])
        selected_tstarts["persample_var"].append(var_win[0])
        selected_tstarts["onset_fixed8"].append(on8_win[0])

        cond_windows={
            "baseline":None,"sweep_best":sweep_win,
            "persample_cos8":ps8_win,"persample_var":var_win,
            "onset_fixed8":on8_win,"calib_onset8":calib_onset8_win,
            "calib_global8":calib_global8_win,
        }
        for cname in COND_NAMES:
            win=cond_windows[cname]
            if win is None:
                pred=mc_predict(model,tok,item)
            else:
                ts,te=win
                n_t_c=te-ts; n_d_c=n_layers-te
                if n_d_c<1 or n_t_c<1: pred=mc_predict(model,tok,item)
                else: pred=mc_predict(model,tok,item,n_e=ts,n_t=n_t_c,k=K_ETD)
            if pred==label: correct[cname]+=1

        if (i+1)%10==0:
            el=time.time()-t0; eta=el/(i+1)*(n_total-i-1)
            line=f"  [{i+1:3d}/{n_total}] "
            for cn in COND_NAMES: line+=f"{cn[:4]}={correct[cn]/(i+1):.3f} "
            line+=f"| {el:.0f}s ETA {eta:.0f}s"
            print(line)
        torch.cuda.empty_cache()

    elapsed=time.time()-t0
    accuracies={c:correct[c]/n_total for c in COND_NAMES}
    win_stats={c:{"t_start_mean":float(np.mean(selected_tstarts[c])) if selected_tstarts[c] else 0.0,
                  "t_start_std":float(np.std(selected_tstarts[c])) if selected_tstarts[c] else 0.0,
                  "t_start_list":selected_tstarts[c]} for c in selected_tstarts}
    return {
        "benchmark":"LogiQA","n":n_total,"elapsed_s":elapsed,
        "accuracies":accuracies,"sweep_best_window":list(sweep_win),
        "calib_global8_window":list(calib_global8_win),
        "calib_onset8_window":list(calib_onset8_win),
        "mean_profile":{str(k):v for k,v in sorted(mean_profile.items())},
        "window_stats":win_stats,
    }


# ─── 最终可视化（8 benchmark 完整版）──────────────────────────────────────────
SIGNAL_CONDS = ["persample_cos8","persample_var","onset_fixed8","calib_onset8","calib_global8"]

def plot_all_benchmark_bars(all_results):
    bench_list=list(all_results.keys())
    n=len(bench_list)
    fig,axes=plt.subplots(2,4,figsize=(22,9)) if n==8 else plt.subplots(2,4,figsize=(22,9))
    axes_flat=axes.flatten()
    for ax_idx,(bench,ax) in enumerate(zip(bench_list,axes_flat)):
        accs=all_results[bench]["accuracies"]
        base_acc=accs["baseline"]; sweep_acc=accs["sweep_best"]
        conds=[c for c in COND_NAMES if c in accs]
        x=np.arange(len(conds))
        bars=[accs[c] for c in conds]
        colors=[COND_COLORS[c] for c in conds]
        brs=ax.bar(x,bars,color=colors,edgecolor="black",linewidth=0.4,alpha=0.88)
        for bar,v in zip(brs,bars):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                    f"{v:.2f}",ha="center",va="bottom",fontsize=6)
        ax.axhline(base_acc,color="grey",linestyle="--",linewidth=1,alpha=0.7)
        ax.axhline(sweep_acc,color="#2196F3",linestyle=":",linewidth=1.5,alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS.get(c,c)[:4] for c in conds],fontsize=6,rotation=40,ha="right")
        best_sig=max(SIGNAL_CONDS,key=lambda c:accs.get(c,-99) if c in accs else -99)
        best_acc=accs.get(best_sig,0)
        delta=best_acc-base_acc
        ax.set_title(f"{bench[:18]}\nbase={base_acc:.3f}→best={best_acc:.3f}({delta:+.3f})",
                     fontsize=7.5)
        ax.set_ylim(0,max(bars)*1.2+0.02)
    for ax in axes_flat[len(bench_list):]: ax.set_visible(False)
    # 统一图例
    from matplotlib.patches import Patch
    legend_handles=[Patch(color=COND_COLORS[c],label=COND_LABELS[c]) for c in COND_NAMES]
    legend_handles+=[plt.Line2D([0],[0],color="grey",linestyle="--",label="Baseline"),
                     plt.Line2D([0],[0],color="#2196F3",linestyle=":",label="扫参最优")]
    fig.legend(handles=legend_handles,loc="lower center",ncol=5,fontsize=8,
               bbox_to_anchor=(0.5,-0.02))
    fig.suptitle("R38 信号引导 ETD：全 Benchmark 条形图（含 R38b 教训：min_start≥9 关键）",
                 fontsize=11,y=1.01)
    plt.tight_layout()
    fname=FIGURES_DIR/"all_benchmark_bars.png"
    plt.savefig(fname,dpi=120,bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")

def plot_heatmap_final(all_results):
    bench_list=list(all_results.keys())
    delta_mat=np.zeros((len(bench_list),len(SIGNAL_CONDS)))
    acc_mat=np.zeros_like(delta_mat)
    for bi,bench in enumerate(bench_list):
        accs=all_results[bench]["accuracies"]
        base=accs["baseline"]
        for ci,cn in enumerate(SIGNAL_CONDS):
            if cn in accs:
                delta_mat[bi,ci]=accs[cn]-base
                acc_mat[bi,ci]=accs[cn]
    vmax=max(np.abs(delta_mat).max(),0.05)
    fig,ax=plt.subplots(figsize=(10,max(5,len(bench_list)*0.9)))
    im=ax.imshow(delta_mat,cmap="RdYlGn",vmin=-vmax,vmax=vmax,aspect="auto")
    plt.colorbar(im,ax=ax,label="Δacc vs Baseline")
    ax.set_xticks(range(len(SIGNAL_CONDS)))
    ax.set_xticklabels([COND_LABELS.get(c,c) for c in SIGNAL_CONDS],fontsize=9,rotation=25,ha="right")
    ax.set_yticks(range(len(bench_list)))
    ax.set_yticklabels(bench_list,fontsize=9)
    for bi in range(len(bench_list)):
        for ci in range(len(SIGNAL_CONDS)):
            delta=delta_mat[bi,ci]; acc=acc_mat[bi,ci]
            color="white" if abs(delta)>vmax*0.5 else "black"
            ax.text(ci,bi,f"{acc:.3f}\n({delta:+.3f})",ha="center",va="center",fontsize=7.5,color=color)
    ax.set_title("R38 热力图：信号方法 Accuracy 及 Δacc vs Baseline（8 benchmarks）",fontsize=11)
    plt.tight_layout()
    fname=FIGURES_DIR/"final_heatmap.png"
    plt.savefig(fname,dpi=120,bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")

def plot_delta_scatter_final(all_results):
    bench_list=list(all_results.keys())
    all_deltas=[]
    for bench in bench_list:
        accs=all_results[bench]["accuracies"]
        base=accs["baseline"]
        for c in SIGNAL_CONDS: all_deltas.append(accs.get(c,base)-base)
        all_deltas.append(accs["sweep_best"]-base)
    lim=max(abs(min(all_deltas)),abs(max(all_deltas)),0.06)+0.02
    fig,ax=plt.subplots(figsize=(7,7))
    ax.plot([-lim,lim],[-lim,lim],"k--",lw=1,alpha=0.5,label="与扫参相当")
    ax.axhline(0,color="grey",lw=0.8,ls=":")
    ax.axvline(0,color="grey",lw=0.8,ls=":")
    for cn in SIGNAL_CONDS:
        xs,ys,lbs=[],[],[]
        for bench in bench_list:
            accs=all_results[bench]["accuracies"]
            base=accs["baseline"]
            xs.append(accs["sweep_best"]-base)
            ys.append(accs.get(cn,base)-base)
            lbs.append(bench[:5])
        ax.scatter(xs,ys,label=COND_LABELS.get(cn,cn),alpha=0.8,s=70,
                   color=COND_COLORS.get(cn,"gray"),edgecolors="black",linewidths=0.5)
        for xi,yi,lb in zip(xs,ys,lbs):
            ax.annotate(lb,(xi,yi),fontsize=6.5,ha="center",va="bottom",
                        xytext=(0,3),textcoords="offset points")
    ax.set_xlabel("扫参最优 Δacc vs Baseline",fontsize=10)
    ax.set_ylabel("信号方法 Δacc vs Baseline",fontsize=10)
    ax.set_title("R38 Δacc 散点图：各信号方法 vs 扫参最优\n（对角线上=与扫参等价）",fontsize=10)
    ax.legend(fontsize=8,loc="upper left")
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    plt.tight_layout()
    fname=FIGURES_DIR/"final_delta_scatter.png"
    plt.savefig(fname,dpi=120,bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")

def plot_calib_profiles_final(all_results):
    bench_list=list(all_results.keys())
    fig,ax=plt.subplots(figsize=(12,5))
    for bench in bench_list:
        profile_raw=all_results[bench].get("mean_profile",{})
        if not profile_raw: continue
        profile={int(k):v for k,v in profile_raw.items()}
        layers=sorted(profile); vals=[profile[l] for l in layers]
        color=BENCH_COLORS.get(bench,"gray")
        ax.plot(layers,vals,"o-",color=color,lw=1.5,ms=4,alpha=0.85,label=bench[:14])
        onset_win=all_results[bench].get("calib_onset8_window")
        if onset_win:
            ts=onset_win[0]
            val_at=profile.get(ts, profile.get(min(profile,key=lambda l:abs(l-ts)),0))
            ax.axvline(ts,color=color,ls=":",lw=1.1,alpha=0.55)
            ax.annotate(f"{bench[:4]}@{ts}",(ts,val_at),fontsize=6,color=color,
                        xytext=(2,2),textcoords="offset points")
    ax.axhline(0.28,color="gray",ls="--",lw=1,alpha=0.6,label="固定阈值 0.28")
    ax.axvspan(6,9,alpha=0.07,color="red",label="早期层假阳性区（min_start限制前）")
    ax.axhline(0,color="black",lw=0.8,alpha=0.3)
    ax.set_xlabel("Layer",fontsize=10); ax.set_ylabel("Mean cos(Term1, Δh)",fontsize=10)
    ax.set_title("R38 标定 Profile（8 benchmarks）\n红色阴影=min_start=9 所排除的早期假阳性区",fontsize=10)
    ax.legend(fontsize=7.5,loc="upper right",ncol=2)
    ax.set_xlim(4,30)
    plt.tight_layout()
    fname=FIGURES_DIR/"final_calib_profiles.png"
    plt.savefig(fname,dpi=120,bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")

def plot_tstart_violin_final(all_results):
    bench_list=list(all_results.keys())
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,cname in zip(axes,["persample_cos8","persample_var"]):
        data,ticks,sweep_tstarts=[],[],[]
        for bench in bench_list:
            ws=all_results[bench].get("window_stats",{}).get(cname)
            if ws and ws.get("t_start_list"):
                data.append(ws["t_start_list"]); ticks.append(bench[:10])
                sweep_tstarts.append(SWEEP_BEST[bench][0])
            else:
                data.append([12]); ticks.append(bench[:10])
                sweep_tstarts.append(SWEEP_BEST[bench][0])
        if any(len(d)>1 for d in data):
            vp=ax.violinplot(data,positions=range(len(ticks)),showmedians=True,showextrema=True)
            for body in vp["bodies"]: body.set_alpha(0.65); body.set_facecolor(COND_COLORS[cname])
        for bi,(ts,pos) in enumerate(zip(sweep_tstarts,range(len(ticks)))):
            ax.scatter(pos,ts,marker="D",s=55,color="#2196F3",zorder=5,
                       label="扫参最优 t_start" if bi==0 else "")
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks,rotation=30,ha="right",fontsize=8)
        ax.set_ylabel("Selected t_start",fontsize=9)
        ax.set_title(f"{COND_LABELS.get(cname,cname)}\n(蓝◆=扫参最优 t_start)",fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("R38 t_start 选层分布（Violin）",fontsize=11,y=1.02)
    plt.tight_layout()
    fname=FIGURES_DIR/"final_tstart_violin.png"
    plt.savefig(fname,dpi=120,bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")

def plot_summary_final(all_results):
    bench_list=list(all_results.keys()); n_bench=len(bench_list)
    fig,ax=plt.subplots(figsize=(14,5))
    x=np.arange(n_bench); w=0.26
    baselines=[all_results[b]["accuracies"]["baseline"] for b in bench_list]
    sweep_accs=[all_results[b]["accuracies"]["sweep_best"] for b in bench_list]
    best_sig_accs=[]; best_sig_names=[]
    for bench in bench_list:
        accs=all_results[bench]["accuracies"]
        best_c=max(SIGNAL_CONDS,key=lambda c:accs.get(c,-99) if c in accs else -99)
        best_sig_accs.append(accs.get(best_c,accs["baseline"]))
        best_sig_names.append(COND_LABELS.get(best_c,best_c))
    b1=ax.bar(x-w,baselines,w,label="Baseline",color="#9E9E9E",alpha=0.85)
    b2=ax.bar(x,best_sig_accs,w,label="最佳信号方法",color="#F44336",alpha=0.85)
    b3=ax.bar(x+w,sweep_accs,w,label="扫参最优",color="#2196F3",alpha=0.85)
    for bar,v in zip(b1,baselines):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,f"{v:.3f}",ha="center",va="bottom",fontsize=6.5)
    for bar,v,nm in zip(b2,best_sig_accs,best_sig_names):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,f"{v:.3f}\n({nm[:5]})",ha="center",va="bottom",fontsize=6)
    for bar,v in zip(b3,sweep_accs):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,f"{v:.3f}",ha="center",va="bottom",fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([b[:14] for b in bench_list],rotation=20,ha="right",fontsize=8)
    ax.set_ylabel("Accuracy",fontsize=10)
    ax.set_title("R38 最终汇总：Baseline vs 最佳信号方法 vs 扫参最优（8 benchmarks）",fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0,min(1.0,max(max(sweep_accs),max(best_sig_accs))*1.25+0.05))
    plt.tight_layout()
    fname=FIGURES_DIR/"final_summary.png"
    plt.savefig(fname,dpi=120,bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")

def print_final_summary(all_results):
    print("\n"+"="*70)
    print("R38 最终结论（8 benchmarks，含 LogiQA）")
    print("="*70)
    bench_list=list(all_results.keys())
    print(f"\n{'Benchmark':28s} {'base':>6} {'sweep':>6} {'best_sig':>9} {'Δbase':>7} {'%sweep':>7} {'方法':>12}")
    for bench in bench_list:
        accs=all_results[bench]["accuracies"]
        base=accs["baseline"]; sweep=accs["sweep_best"]
        avail=[c for c in SIGNAL_CONDS if c in accs]
        if avail:
            best_c=max(avail,key=lambda c:accs[c])
            best_acc=accs[best_c]; delta=best_acc-base
            pct=best_acc/sweep if sweep>0 else 0.0
            print(f"{bench[:28]:28s} {base:6.3f} {sweep:6.3f} {best_acc:9.3f} {delta:7.3f} {pct:7.1%} {COND_LABELS.get(best_c,best_c)[:12]:>12}")
        else:
            print(f"{bench[:28]:28s} {base:6.3f} {sweep:6.3f}   N/A")

    n_bench=len(bench_list)
    avail_benches=[b for b in bench_list if any(c in all_results[b]["accuracies"] for c in SIGNAL_CONDS)]
    print(f"\n{'─'*70}")
    print(f"{'方法':14s} {'wins':>6} {'beats_base':>12} {'macro_Δ':>9}")
    for cn in SIGNAL_CONDS:
        avail=[b for b in avail_benches if cn in all_results[b]["accuracies"]]
        if not avail: continue
        wins=sum(1 for b in avail if max((all_results[b]["accuracies"].get(c,-99) for c in SIGNAL_CONDS if c in all_results[b]["accuracies"]),default=-99)==all_results[b]["accuracies"].get(cn,-99))
        beats=sum(1 for b in avail if all_results[b]["accuracies"][cn]>all_results[b]["accuracies"]["baseline"])
        macro=np.mean([all_results[b]["accuracies"][cn]-all_results[b]["accuracies"]["baseline"] for b in avail])
        print(f"{COND_LABELS.get(cn,cn):14s} {wins:>6}/{n_bench}  {beats:>6}/{len(avail)}     {macro:+.4f}")
    sweep_macro=np.mean([all_results[b]["accuracies"]["sweep_best"]-all_results[b]["accuracies"]["baseline"] for b in bench_list])
    print(f"{'扫参最优':14s} {'─':>6}       {'─':>6}      {sweep_macro:+.4f}")


def main():
    t0=time.time()
    print("="*70)
    print("R38c: LogiQA 补全 + 最终可视化")
    print("="*70)

    # 加载 R38a 结果
    r38a_path=RESULTS_DIR/"r38_signal_full_bench_results.json"
    final_path=RESULTS_DIR/"r38_final_results.json"

    if final_path.exists():
        with open(final_path) as f: all_results=json.load(f)
        print(f"[恢复] {len(all_results)} benchmarks")
    elif r38a_path.exists():
        with open(r38a_path) as f: all_results=json.load(f)
        print(f"[导入] R38a ({len(all_results)} benchmarks)")
    else:
        all_results={}

    # 检查 LogiQA 是否需要运行
    if "LogiQA" not in all_results or "baseline" not in all_results.get("LogiQA",{}).get("accuracies",{}):
        print("\n补全 LogiQA …")
        tok,model,n_layers=load_model()
        try:
            items=load_logiqa_fixed(100)
            print(f"  LogiQA: 加载 {len(items)} 样本")
            logiqa_result=evaluate_logiqa(items,model,tok,n_layers)
            all_results["LogiQA"]=logiqa_result
            accs=logiqa_result["accuracies"]
            print(f"\n  === LogiQA Results ===")
            for cn in COND_NAMES:
                delta=accs[cn]-accs["baseline"]
                print(f"    {cn:18s}: {accs[cn]:.4f}  Δ={delta:+.4f}  "
                      f"{accs[cn]/accs['sweep_best']:.1%} of sweep" if accs["sweep_best"]>0 else "")
        except Exception as e:
            print(f"  [ERROR] LogiQA: {e}")
        with open(final_path,"w") as f: json.dump(all_results,f,indent=2)
    else:
        print("  [跳过] LogiQA 已有结果")

    # 生成最终可视化
    print("\n生成最终可视化（8 benchmarks）…")
    if all_results:
        plot_all_benchmark_bars(all_results)
        plot_heatmap_final(all_results)
        plot_delta_scatter_final(all_results)
        plot_calib_profiles_final(all_results)
        plot_tstart_violin_final(all_results)
        plot_summary_final(all_results)

    print_final_summary(all_results)

    with open(final_path,"w") as f: json.dump(all_results,f,indent=2)
    print(f"\nSaved → {final_path}")
    print(f"\nR38c 完成！耗时 {time.time()-t0:.0f}s")


if __name__=="__main__":
    main()
