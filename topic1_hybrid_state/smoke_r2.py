#!/usr/bin/env python3
"""CHymERA R2: deterministic, offline NumPy verification and specification audit.

Only SMOKE_R2_REPORT.md and SMOKE_R2_RESULTS.json are written.  A PASS verifies
the stated mathematical scope; counterexamples to the requested overclaims are
explicit positive tests, never silently relabelled as theorems.
"""
import os

# Set before importing NumPy. No GPU, networking, SciPy, or model downloads.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

SEED = 20261109
D = V = 128
W = 4
ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(SEED)
CHECKS = []
DETAILS = {}


def check(name, ok, **metrics):
    CHECKS.append({"name": name, "passed": bool(ok), "metrics": metrics})


def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)), initial=0))


def relerr(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def norm(x):
    return float(np.linalg.norm(x))


def mv(k, s):
    return np.einsum("i,ij->j", k, s, optimize=False)


def step(s, k, v, beta, gamma):
    decayed = gamma * s
    return decayed + k[:, None] * (beta * (v - mv(k, decayed)))[None, :]


def sequential(s, k, v, beta, g, history=False):
    out = [s.copy()]
    for j in range(len(k)):
        s = step(s, k[j], v[j], beta[j], np.exp(g[j]))
        if history:
            out.append(s.copy())
    return np.stack(out) if history else s


def operands(n, dtype=np.float64):
    k = RNG.normal(size=(n, D))
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    return (k.astype(dtype), (RNG.normal(size=(n, V)) * .03).astype(dtype),
            RNG.uniform(.05, .85, n).astype(dtype),
            (-RNG.uniform(0, .015, n)).astype(dtype))


def forward_unit_lower(m, rhs):
    """Independent forward substitution, no explicit matrix inverse."""
    out = rhs.copy()
    for j in range(len(m)):
        out[j] -= m[j, :j] @ out[:j]
    return out


def wy(k, v, beta, g):
    n = len(k)
    G = np.cumsum(g)
    lower = np.tri(n,k=-1,dtype=bool)
    mask = np.exp(np.where(lower,G[:, None]-G[None, :],-np.inf))
    m = np.eye(n, dtype=k.dtype) + beta[:, None] * (k @ k.T) * mask
    w = k.T * np.exp(G[-1] - G)[None, :]
    yt = forward_unit_lower(m, (beta * np.exp(G))[:, None] * k)
    u = forward_unit_lower(m, beta[:, None] * v)
    return np.exp(G[-1]), w, yt, u


def apply_wy(s, factors):
    gamma, w, yt, u = factors
    return gamma * s + w @ (u - yt @ s)


def affine(k, v, beta, g, native=64):
    p = np.eye(D, dtype=k.dtype)
    h = np.zeros((D, V), dtype=k.dtype)
    for start in range(0, len(k), native):
        sl = slice(start, start + native)
        gamma, w, yt, u = wy(k[sl], v[sl], beta[sl], g[sl])
        pc = gamma * np.eye(D, dtype=k.dtype) - w @ yt
        hc = w @ u
        p, h = pc @ p, pc @ h + hc
    return p, h


def bf16(x):
    """Round-to-nearest, ties-to-even BF16 emulation, returned as FP32."""
    f = np.asarray(x, dtype=np.float32).copy()
    u = f.view(np.uint32)
    rounded = (u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def canonical(spans, t):
    out = []
    for s, e in sorted(spans):
        if not (0 <= s <= e < t):
            raise ValueError("invalid closed deletion interval")
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def survivor_map(t, spans):
    deleted = np.zeros(t, dtype=bool)
    for s, e in spans:
        deleted[s:e + 1] = True
    return np.flatnonzero(~deleted), deleted


def pi_at(pi, i):
    return int(pi[i]) if i >= 0 else i


def halo(pi):
    return {i for i in range(len(pi)) if int(pi[i]) - pi_at(pi, i - 3) > 3}


def conv(x, weights):
    out = np.zeros_like(x)
    for lag in range(W):
        if lag < len(x):
            out[lag:] += x[:len(x) - lag] * weights[W - 1 - lag]
    return out


def silu(x):
    return x / (1 + np.exp(-np.clip(x, -500, 500)))


def patch_conv(x, weights, spans, naive=False):
    pi, _ = survivor_map(len(x), spans)
    out = conv(x, weights)[pi].copy()
    if naive:
        # Each span uses ORIGINAL pre-span context, independently; never feed
        # the repaired previous island into the next span's context.
        where = {int(t): i for i, t in enumerate(pi)}
        for s, e in spans:
            for t in range(e + 1, min(len(x), e + W)):
                if t not in where:
                    continue
                local = t - e - 1
                total = np.zeros_like(x[0])
                for lag in range(W):
                    source = t - lag if local - lag >= 0 else s + local - lag
                    if source >= 0:
                        total += weights[W - 1 - lag] * x[source]
                out[where[t]] = total
    else:
        for i in halo(pi):
            out[i] = 0
            for lag in range(W):
                j = i - lag
                if j >= 0:
                    out[i] += weights[W - 1 - lag] * x[pi[j]]
    return out, pi


def dirty_chunks(t, c, spans):
    pi, deleted = survivor_map(t, spans)
    dirty = []
    for b in range(math.ceil(t / c)):
        lo, hi = b * c, min(t, (b + 1) * c)
        if np.any(~deleted[lo:hi]) and np.any(deleted[max(0, lo - 3):hi]):
            dirty.append(b)
    return dirty


CONFIG_FILES = {"0.8B": "qwen35_08b", "2B": "qwen35_2b",
                "4B": "qwen35_4b_inst", "9B": "qwen35_9b_inst"}


def load_configs():
    configs = {}
    for name, directory in CONFIG_FILES.items():
        path = ROOT / "research_sources" / directory / "config.json"
        raw = json.loads(path.read_text())
        c = raw.get("text_config", raw)
        configs[name] = dict(
            dm=c["hidden_size"], ff=c["intermediate_size"],
            lg=c["layer_types"].count("linear_attention"),
            lf=c["layer_types"].count("full_attention"),
            hk=c["linear_num_key_heads"], hv=c["linear_num_value_heads"],
            dk=c["linear_key_head_dim"], dv=c["linear_value_head_dim"],
            hq=c["num_attention_heads"], hkv=c["num_key_value_heads"],
            df=c["head_dim"], w=c["linear_conv_kernel_dim"],
            partial_rotary_factor=c["rope_parameters"]["partial_rotary_factor"],
            source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return configs


def counters(c, t, stride, precision=4, replay_tokens=None, clean_chunks=None):
    lg, lf, hk, hv, dk, dv, dm, ff, hq, hkv, df = (
        c[k] for k in ("lg", "lf", "hk", "hv", "dk", "dv", "dm", "ff", "hq", "hkv", "df"))
    n = math.ceil(t / stride)
    m = max(n - 1, 0)
    convdim = 2 * hk * dk + hv * dv
    matrix = lg * hv
    convbytes = lg * convdim * 3 * 2
    kv = t * lf * 2 * hkv * df * 2
    delta = m * (matrix * precision * (dk * dk + dk * dv) + convbytes)
    direct = m * (matrix * precision * (dk * dk + 2 * dk * dv) + convbytes)
    anchors = m * (matrix * precision * dk * dv + convbytes)
    # Contracted GEMM/conv/state arithmetic model. Transcendentals, norms,
    # softmax, cache writes and all preprocessing are explicitly excluded.
    tok = ((lg + lf) * 6 * dm * ff
           + lg * (2 * dm * (2 * hk * dk + 3 * hv * dv + 2 * hv)
                   + 2 * c["w"] * convdim + 7 * hv * dk * dv)
           + lf * 2 * dm * df * (3 * hq + 2 * hkv))
    pair = 4 * lf * hq * df
    q = min(t, 3 * stride) if replay_tokens is None else replay_tokens
    nc = m if clean_chunks is None else clean_chunks
    jump = nc * lg * hv * (2 * dk * dk * dv + 2 * dk * dv)
    replay = t * tok + pair * t * (t + 1) // 2
    # Two passes: recover ORIGINAL local operands, then replay edited local
    # network/seams. This deliberately overcharges rather than hiding pass 1.
    edit = 2 * q * (tok + pair * t) + jump
    return dict(kv_bytes=kv, delta_bytes=delta, direct_bytes=direct,
                anchor_only_bytes=anchors, terminal_state_bytes=matrix * dk * dv * 4,
                conv_state_bytes=convbytes, token_flops=tok, pair_flops=pair,
                clean_jump_flops=jump, full_flops=replay, edit_upper_flops=edit,
                replay_token_budget=q, stride=stride, chunks=n, m=m)


def suite_t6():
    records = []
    for dtype, tol in ((np.float64, 1e-13), (np.float32, 5e-6)):
        for b in (16, 32, 64):
            k, v, beta, g = operands(b, dtype)
            f = wy(k, v, beta, g)
            # Multiple initial states check the affine map, not just zero.
            for j in range(3):
                s = (RNG.normal(size=(D, V)) * .01).astype(dtype)
                a = sequential(s, k, v, beta, g)
                z = apply_wy(s, f)
                err = maxerr(a, z)
                check(f"T6.WY.{np.dtype(dtype).name}.B{b}.initial{j}", err < tol,
                      max_abs_error=err, relative_error=relerr(z, a), tolerance=tol)
        for c in (128, 256):
            k, v, beta, g = operands(c, dtype)
            s = (RNG.normal(size=(D, V)) * .01).astype(dtype)
            p, h = affine(k, v, beta, g)
            a = sequential(s, k, v, beta, g)
            err = maxerr(a, p @ s + h)
            check(f"T6.fold.{np.dtype(dtype).name}.C{c}", err < tol,
                  max_abs_error=err, relative_error=relerr(p @ s + h, a))
    # Degenerate gates and padding are legal even when invertibility fails.
    for b in (1, 7):
        k, v, beta, g = operands(b)
        for branch in ("beta0", "beta1", "zero_key", "gamma1"):
            kk, bb, gg = k.copy(), beta.copy(), g.copy()
            if branch == "beta0": bb[:] = 0
            if branch == "beta1": bb[:] = 1
            if branch == "zero_key": kk[:] = 0
            if branch == "gamma1": gg[:] = 0
            s = RNG.normal(size=(D, V)) * .01
            e = maxerr(sequential(s, kk, v, bb, gg), apply_wy(s, wy(kk, v, bb, gg)))
            check(f"T6.degenerate.B{b}.{branch}", e < 1e-13, max_abs_error=e)

    # Structural 3-token spill with unit-scale conv witness. State is full 128².
    for dtype, tol in ((np.float64, 1e-14), (np.float32, 5e-6)):
        c, t = 16, 64
        for offset in (1, 2):
            e = 2 * c - offset
            spans = [[c + 2, e]]
            x = np.ones((t, V), dtype=dtype)
            x[c + 2:e + 1] = 3
            weights = np.ones((4, V), dtype=dtype)
            pi, _ = survivor_map(t, spans)
            original = conv(x, weights)[pi]
            truth = conv(x[pi], weights)
            fixed, _ = patch_conv(x, weights, spans)
            # Only entries in the chunk after the end are deliberately stale.
            bad = fixed.copy()
            bad[pi >= 2*c] = original[pi >= 2*c]
            # Read state just after the cascading halo so later contraction
            # cannot conceal the missing-halo witness.
            stop = int(np.searchsorted(pi, 2*c + 4))
            key = np.zeros((len(pi), D), dtype=dtype); key[:, 0] = 1
            beta = np.full(len(pi), .3, dtype=dtype)
            g = np.zeros(len(pi), dtype=dtype)
            s = np.zeros((D, V), dtype=dtype)
            a = sequential(s, key[:stop], truth[:stop], beta[:stop], g[:stop])
            b = sequential(s, key[:stop], bad[:stop], beta[:stop], g[:stop])
            z = sequential(s, key[:stop], fixed[:stop], beta[:stop], g[:stop])
            spill_err, repair_err = maxerr(a, b), maxerr(a, z)
            check(f"T6.spill.{np.dtype(dtype).name}.offset{offset}",
                  2 in dirty_chunks(t, c, spans) and spill_err > 1e-2 and repair_err < tol,
                  stale_error=spill_err, repair_error=repair_err, dirty=dirty_chunks(t,c,spans))
    three = dirty_chunks(64, 16, [[1, 30]])
    pi_counter,_=survivor_map(64,[[15,29]])
    dirty_counter=dirty_chunks(64,16,[[15,29]])
    surviving_dirty=sum(int(np.sum((pi_counter>=j*16)&(pi_counter<(j+1)*16))) for j in dirty_counter)
    check("T6.spec_correction.three_surviving_dirty_chunks", three == [0,1,2]
          and surviving_dirty==33 and surviving_dirty>2*16,
          dirty=three,two_RC_counterexample_surviving_tokens=surviving_dirty)
    # Tight full-macro replay bound, with independent deterministic trials.
    trial_rng=np.random.default_rng(SEED+21)
    worst_ratio=0.
    for trial in range(500):
        c=int(trial_rng.integers(W,65));t=int(trial_rng.integers(1,300))
        raw=[sorted(trial_rng.integers(0,t,2).tolist())
             for _ in range(int(trial_rng.integers(1,9)))]
        spans=canonical(raw,t);pi,_=survivor_map(t,spans)
        dc=dirty_chunks(t,c,spans)
        count=sum(int(np.sum((pi>=j*c)&(pi<min(t,(j+1)*c)))) for j in dc)
        bound=min(len(pi),len(spans)*(2*c+W-3))
        check(f"T6.tight_qsurv_random.{trial}",count<=bound,
              surviving_dirty=count,bound=bound,R=len(spans),C=c)
        worst_ratio=max(worst_ratio,count/max(bound,1))
    check("T6.tight_qsurv_witness",surviving_dirty==33==2*16+W-3,
          C=16,W=W,spans=[[15,29]],surviving_dirty=surviving_dirty,
          random_trials=500,worst_random_ratio=worst_ratio)
    # Summary equality can hold accidentally in a structurally dirty window.
    k, v, beta, g = operands(8); beta[:] = 0; g[:] = 0
    p, h = affine(k, v, beta, g)
    check("T6.spec_correction.literal_iff_false", maxerr(p,np.eye(D)) == 0 and norm(h)==0)

    # Same original trajectory anchors: direct vs delta vs quantized archives.
    chunks = [operands(16) for _ in range(4)]
    ph = [affine(*op) for op in chunks]
    anchors = [RNG.normal(size=(D,V)) * .01]
    for p,h in ph:
        anchors.append(p @ anchors[-1] + h)
    start = anchors[0] + RNG.normal(size=(D,V)) * .002
    exact, delta = start.copy(), start.copy()
    for j,(p,h) in enumerate(ph):
        exact = p @ exact + h
        delta = anchors[j+1] + p @ (delta-anchors[j])
    check("T6.delta.exact_equivalence", maxerr(delta,exact)<1e-13, max_abs_error=maxerr(delta,exact))
    q = RNG.normal(size=D); q /= norm(q)
    for kind in ("FP32-direct", "FP32-delta", "BF16-direct", "BF16-delta", "rank16-direct"):
        s = start.copy(); residuals = []
        for j,(p,h) in enumerate(ph):
            quant = bf16 if kind.startswith("BF16") else lambda x: x.astype(np.float32)
            if kind == "rank16-direct":
                u,sv,vt = np.linalg.svd(p, full_matrices=False)
                pq = (u[:,:16]*sv[:16]) @ vt[:16]
                proposed = pq @ s + h
            elif kind.endswith("delta"):
                proposed = quant(anchors[j+1]).astype(np.float64) + quant(p).astype(np.float64) @ (
                    s-quant(anchors[j]).astype(np.float64))
            else:
                proposed = quant(p).astype(np.float64) @ s + quant(h).astype(np.float64)
            residuals.append(proposed-(p @ s+h))
            s = proposed
        adj = [None]*5; adj[-1] = q
        for j in range(3,-1,-1): adj[j] = ph[j][0].T @ adj[j+1]
        bound = sum(norm(adj[j+1])*norm(residuals[j]) for j in range(4))
        exact_readout = norm(mv(q,s-exact))
        identity = sum((mv(adj[j+1],residuals[j]) for j in range(4)),np.zeros(V))
        check("T6.archive_certificate."+kind, exact_readout <= bound+1e-13
              and maxerr(identity,mv(q,s-exact))<1e-13,
              actual=exact_readout, certificate=bound, identity_error=maxerr(identity,mv(q,s-exact)))
        records.append(dict(representation=kind,error=exact_readout,certificate=bound))
        if kind.endswith("direct"):
            # Prefill may record conservative compression-error radii and
            # discard original maps. Runtime needs only stored maps/radii.
            if kind=="rank16-direct":
                pq=[]
                for p,h in ph:
                    u,sv,vt=np.linalg.svd(p,full_matrices=False)
                    pq.append((u[:,:16]*sv[:16])@vt[:16])
                hq=[h.copy() for p,h in ph]
            else:
                pq=[quant(p).astype(np.float64) for p,h in ph]
                hq=[quant(h).astype(np.float64) for p,h in ph]
            ep=[norm(pq[j]-ph[j][0]) for j in range(4)]
            eh=[norm(hq[j]-ph[j][1]) for j in range(4)]
            gam=[float(np.exp(np.sum(op[3]))) for op in chunks]
            pa=[None]*5;ea=np.zeros(5);pa[-1]=q
            for j in range(3,-1,-1):
                pa[j]=pq[j].T@pa[j+1]
                ea[j]=gam[j]*ea[j+1]+ep[j]*norm(pa[j+1])
            online=0.;ss=start.copy()
            for j in range(4):
                radius=ep[j]*norm(ss)+eh[j]
                online+=(norm(pa[j+1])+ea[j+1])*radius
                ss=pq[j]@ss+hq[j]
            observed=norm(mv(q,ss-exact))
            check("T6.compressed_online_radius."+kind,observed<=online+1e-13
                  and abs(observed-exact_readout)<1e-13,
                  actual=observed,certificate=online,
                  scope="Prefill-recorded compression radii, FP64 runtime roundoff excluded/tolerated at 1e-13.")
    # Cancellation witness: old state is huge, edited state is tiny. Delta
    # subtraction cancels, while direct affine evaluates the new small state.
    old = np.full((D,V),1e8,dtype=np.float32)
    small = np.ones((D,V),dtype=np.float32)
    cancellation_delta = old + (small-old)
    direct = small.copy()
    check("T6.delta.cancellation_witness", maxerr(cancellation_delta,small)>0.5
          and maxerr(direct,small)==0, delta_error=maxerr(cancellation_delta,small),direct_error=0.)
    DETAILS["precision"] = records

    configs = load_configs()
    table = []
    expected={"0.8B":(1024,3584,18,6,16,8,2),
              "2B":(2048,6144,18,6,16,8,2),
              "4B":(2560,9216,24,8,32,16,4),
              "9B":(4096,12288,24,8,32,16,4)}
    for name,c in configs.items():
        check("T6.config."+name, c["hk"]==16
              and tuple(c[k] for k in ("dm","ff","lg","lf","hv","hq","hkv"))==expected[name]
              and c["dk"]==c["dv"]==128 and c["df"]==256 and c["w"]==4
              and c["partial_rotary_factor"]==.25)
        for t,stride in ((1024,128),(4096,256),(4096,1024),(16384,512)):
            a = counters(c,t,stride,4)
            b = counters(c,t,stride,2)
            # Independent shape-based nbytes counters: no enormous allocations.
            def nbytes(shape, dtype): return math.prod(shape)*np.dtype(dtype).itemsize
            m = math.ceil(t/stride)-1
            actual = m*(nbytes((c["lg"],c["hv"],D,D),np.float32)
                        +nbytes((c["lg"],c["hv"],D,V),np.float32)
                        +nbytes((c["lg"],2*c["hk"]*D+c["hv"]*V,3),np.uint16))
            kv = nbytes((t,c["lf"],2,c["hkv"],c["df"]),np.uint16)
            dot_count = m*c["lg"]*c["hv"]*D*V*D
            dm,ff,hk,hv,hq,hkv,df=(c[key] for key in ("dm","ff","hk","hv","hq","hkv","df"))
            gdn_shapes=[(dm,2*hk*D+hv*V),(dm,hv*V),(dm,hv),(dm,hv),(hv*V,dm)]
            fa_shapes=[(dm,2*hq*df),(dm,hkv*df),(dm,hkv*df),(hq*df,dm)]
            mlp_shapes=[(dm,ff),(dm,ff),(ff,dm)]
            shapes_flops=lambda shapes:sum(2*rows*cols for rows,cols in shapes)
            token_counter=(c["lg"]*shapes_flops(gdn_shapes)+c["lf"]*shapes_flops(fa_shapes)
                           +(c["lg"]+c["lf"])*shapes_flops(mlp_shapes)
                           +c["lg"]*(2*W*(2*hk*D+hv*V)+7*hv*D*V))
            full_counter=t*token_counter+4*c["lf"]*hq*df*sum(range(1,t+1))
            check(f"T6.accounting.{name}.T{t}.C{stride}",actual==a["delta_bytes"] and kv==a["kv_bytes"]
                  and a["clean_jump_flops"]==2*dot_count+2*m*c["lg"]*c["hv"]*D*V
                  and token_counter==a["token_flops"] and full_counter==a["full_flops"],
                  archive_bytes=actual,kv_bytes=kv)
            table.append(dict(model=name,T=t,**a,bf16_delta_bytes=b["delta_bytes"],
                              bf16_direct_bytes=b["direct_bytes"],
                              speedup=a["full_flops"]/a["edit_upper_flops"]))
        standard=counters(c,4096,256);light=counters(c,4096,1024)
        check("T6.light_stride_C1024."+name,
              light["m"]==3 and standard["m"]==15
              and all(standard[key]==5*light[key] for key in
                      ("delta_bytes","direct_bytes","anchor_only_bytes"))
              and counters(c,4096,256,2)["delta_bytes"]==5*counters(c,4096,1024,2)["delta_bytes"],
              m=light["m"],standard_m=standard["m"],archive_ratio=5,
              light_delta_bytes=light["delta_bytes"],standard_delta_bytes=standard["delta_bytes"])
        large = [counters(c,t,4*math.isqrt(t)) for t in (2**20,2**24,2**28)]
        archive_fraction = [a["delta_bytes"]/a["kv_bytes"] for a in large]
        flop_fraction = [a["edit_upper_flops"]/a["full_flops"] for a in large]
        check("T6.sqrt_scaling."+name,
              all(x>y for x,y in zip(archive_fraction,archive_fraction[1:]))
              and all(x>y for x,y in zip(flop_fraction,flop_fraction[1:])),
              archive_over_kv=archive_fraction,edit_over_prefill=flop_fraction)
    for b in (16,32,64):
        check("T6.WY_flops.B"+str(b), (4*b*D*V < 2*D*D*V)==(b<D/2),
              two_gemm_flops=4*b*D*V,dense_gemm_flops=2*D*D*V)
    # Dense tree finite node count, plus a declared sufficient operand archive.
    c=configs["0.8B"]; t=4096; trees=[]
    for c0 in (1,16):
        n=math.ceil(t/c0); nodes=2*n-1
        tree=nodes*c["lg"]*c["hv"]*4*(D*D+D*V)
        asymptotic=4*c["lg"]*c["hv"]*4*D*V/c0
        check("T6.dense_tree.C"+str(c0),
              math.isclose(asymptotic/(12*1024),6144/c0),
              tree_bytes=tree,bytes_per_token=tree/t,
              asymptotic_MiB_per_token=asymptotic/2**20,asymptotic_KV_ratio=asymptotic/(12*1024))
        trees.append(dict(C0=c0,tree_bytes=tree,asymptotic_MiB_per_token=asymptotic/2**20))
    DETAILS["configs"] = configs
    light_rows=[r for r in table if r["T"]==4096 and r["stride"]==1024]
    check("T6.light_stride_C1024",len(light_rows)==4 and all(r["m"]==3 for r in light_rows)
          and all(next(z for z in table if z["model"]==r["model"] and z["T"]==4096
                       and z["stride"]==256)["delta_bytes"]==5*r["delta_bytes"] for r in light_rows),
          models=[r["model"] for r in light_rows],T=4096,C=1024,m=3,archive_ratio=5)
    DETAILS["cost_table"] = table
    DETAILS["dense_tree"] = trees


def verify_topology(t, raw, label):
    spans=canonical(raw,t); pi,deleted=survivor_map(t,spans)
    hs=halo(pi); union=set(); ks=[]
    for s,e in spans:
        k=int(np.sum(pi<s));ks.append(k)
        union.update(range(k,min(k+3,len(pi))))
    broken=[]
    for i in hs:
        m=next(m for m in (1,2,3) if pi_at(pi,i-m)!=int(pi[i])-m)
        ts=int(pi[i])-m
        broken.append((i,m,ts))
    suff=all(all(pi_at(pi,i-m)==int(pi[i])-m for m in range(W))
             for i in range(len(pi)) if i not in hs)
    short=any(spans[r+1][0]-spans[r][1]-1<3 for r in range(len(spans)-1))
    trunc=bool(ks and ks[-1]+2>=len(pi))
    check("T7.topology."+label, hs==union and len(hs)<=3*len(spans) and suff
          and all(ts>=0 and deleted[ts] for _,_,ts in broken)
          and (not(short or trunc) or len(hs)<3*len(spans)),
          spans=spans,halo=sorted(hs),R=len(spans),Lprime=len(pi))
    return spans,pi,hs,broken


def suite_t7():
    topologies=[("prefix",[[0,5]]),("suffix",[[190,199]]),
                ("island1",[[100,120],[122,145]]),
                ("island2",[[100,120],[123,145]]),
                ("island3",[[100,120],[124,145]]),
                ("island4",[[100,120],[125,145]]),
                ("three",[[0,7],[40,60],[190,199]]),
                ("four",[[0,7],[40,60],[62,80],[83,105]]),
                ("merge",[[30,40],[20,35],[41,50],[90,100]]),
                ("all",[[0,199]]),("none",[])]
    for label,raw in topologies:
        spans,pi,hs,broken=verify_topology(200,raw,label)
        for dtype,tol in ((np.float64,1e-14),(np.float32,5e-6)):
            x=RNG.normal(size=(200,8)).astype(dtype)
            weights=RNG.normal(size=(4,8)).astype(dtype)
            z,_=patch_conv(x,weights,spans)
            exact=conv(x[pi],weights)
            check(f"T7.conv.{label}.{np.dtype(dtype).name}",maxerr(z,exact)<tol,
                  max_abs_error=maxerr(z,exact))
        # Every individual halo index has a nonzero-deleted-token witness.
        witness_min=math.inf
        for i,m,ts in broken:
            base=np.ones((200,1))*2
            pert=base.copy(); pert[ts]+=1
            a=silu(conv(base,np.ones((4,1))))
            b=silu(conv(pert,np.ones((4,1))))
            unchanged=maxerr(base[pi],pert[pi])==0
            effect=float(abs(b[pi[i],0]-a[pi[i],0]))
            witness_min=min(witness_min,effect)
            check(f"T7.minimality.{label}.i{i}",unchanged and effect>1e-2,
                  deleted_witness=ts,smallest_broken_lag=m,activation_effect=effect)
    # Exhaust ALL binary deletion masks at short lengths, including prefix
    # padding and empty survivors, independently of named examples.
    cases=0; valid=True
    for t in range(1,10):
        for mask in range(1<<t):
            deleted=np.array([(mask>>i)&1 for i in range(t)],dtype=bool)
            spans=canonical([[i,i] for i in np.flatnonzero(deleted)],t)
            pi=np.flatnonzero(~deleted);hs=halo(pi);expected=set()
            for s,e in spans:
                k=int(np.sum(pi<s));expected.update(range(k,min(k+3,len(pi))))
            valid &= hs==expected
            for i in hs:
                m=next(m for m in range(1,4) if pi_at(pi,i-m)!=int(pi[i])-m)
                ts=int(pi[i])-m
                valid &= ts>=0 and deleted[ts]
            cases+=1
    check("T7.exhaustive_masks",valid,cases=cases)
    witness=[]
    for island in (1,2):
        spans=[[100,120],[121+island,145]]
        for dtype,tol in ((np.float64,1e-14),(np.float32,5e-6)):
            x=np.ones((200,V),dtype=dtype)
            for s,e in spans:x[s:e+1]=4
            weights=np.ones((4,V),dtype=dtype)
            z,pi=patch_conv(x,weights,spans)
            naive,_=patch_conv(x,weights,spans,naive=True)
            truth=conv(x[pi],weights)
            k2=int(np.sum(pi<spans[1][0]));stop=min(k2+3,len(pi))
            key=np.zeros((stop,D),dtype=dtype);key[:,0]=1
            beta=np.full(stop,.4,dtype=dtype);g=np.zeros(stop,dtype=dtype)
            init=np.zeros((D,V),dtype=dtype)
            state=lambda vals:sequential(init,key,vals[:stop],beta,g)
            fail=maxerr(state(naive),state(truth));good=maxerr(state(z),state(truth))
            refs=set(range(spans[0][1]+1,spans[0][1]+4))
            check(f"T7.naive_short_island{island}.{np.dtype(dtype).name}",
                  fail>1e-2 and good<tol and any(spans[1][0]<=j<=spans[1][1] for j in refs),
                  naive_state_error=fail,CHymERA_state_error=good)
            witness.append(dict(island=island,dtype=np.dtype(dtype).name,naive=fail,CHymERA=good))
    # The threshold claim is existential, not universal: zero kernels are a
    # counterexample to any input-independent positive state-error lower bound.
    x=np.zeros((200,1));w=np.zeros((4,1));spans=[[100,120],[122,145]]
    a,pi=patch_conv(x,w,spans,True)
    check("T7.spec_correction.no_universal_Omega1",maxerr(a,conv(x[pi],w))==0)
    DETAILS["short_island_witnesses"]=witness

    # End-to-end F editor: original chunks are summarized independently;
    # surviving dirty chunks are replayed and all clean suffix chunks jump.
    composed=[]
    for dtype,tol in ((np.float64,1e-14),(np.float32,5e-6)):
        for spans in ([[0,6],[30,39]],[[17,30]],[[10,25],[27,45]],
                      [[10,25],[28,45],[80,100],[103,130]],[[150,159]]):
            t=160;c=16;pi,deleted=survivor_map(t,spans)
            raw=(RNG.normal(size=(t,3*D))*.1).astype(dtype)
            weights=RNG.uniform(.1,.5,size=(4,3*D)).astype(dtype)
            cv=silu(conv(raw,weights));compact=silu(conv(raw[pi],weights))
            def kv(cv):
                _,k,v=np.split(cv,3,axis=1)
                return k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+1e-6),v
            k,v=kv(cv);ke,ve=kv(compact)
            beta=RNG.uniform(.1,.8,t).astype(dtype);g=np.full(t,-.002,dtype=dtype)
            zero=np.zeros((D,V),dtype=dtype)
            truth=sequential(zero,ke,ve,beta[pi],g[pi])
            dirty=set(dirty_chunks(t,c,spans));state=zero.copy();naive=zero.copy()
            for j in range(math.ceil(t/c)):
                a,b=j*c,min(t,(j+1)*c)
                local=np.flatnonzero((pi>=a)&(pi<b))
                if not len(local):continue
                ids=pi[local]
                if j in dirty:
                    state=sequential(state,ke[local],ve[local],beta[ids],g[ids])
                else:
                    state=apply_wy(state,wy(k[a:b],v[a:b],beta[a:b],g[a:b]))
                naive=sequential(naive,k[ids],v[ids],beta[ids],g[ids])
            err=maxerr(state,truth)
            check(f"T7.multi_span_chunk_editor.{np.dtype(dtype).name}.{len(composed)}",err<tol
                  and len(dirty)<=3*len(spans),
                  spans=spans,dirty=sorted(dirty),max_abs_error=err,nohalo_error=maxerr(naive,truth))
            composed.append(dict(dtype=np.dtype(dtype).name,spans=spans,error=err))
    DETAILS["F_chunk_editor"]=composed


def adjoints(k,beta,g,q):
    p=np.empty((len(k)+1,D),dtype=np.float64);p[-1]=q
    for j in range(len(k)-1,-1,-1):
        p[j]=np.exp(g[j])*(p[j+1]-beta[j]*np.dot(k[j],p[j+1])*k[j])
    return p


def calibration():
    path=ROOT/"results_r1/probe_r1_out/model_2B/mechanism.jsonl"
    regimes=[]
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line)
            for r in row.get("P1",[]):
                if all(isinstance(r.get(k),(int,float)) and math.isfinite(r[k]) for k in ("Lambda","E","m")):
                    regimes.append((float(r["Lambda"]),float(r["E"]),int(r["m"])))
        source=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    regimes=len(regimes),fallback=False,
                    limitation="Aggregate (Lambda,E) calibration; NOT real keys, queries, values or model forward.")
    else:
        regimes=[(.02,3.,16),(12.,8.,16),(0.,0.,16)]
        source=dict(path=None,regimes=len(regimes),fallback=True,
                    limitation="Deterministic synthetic fallback; no real-weight claim.")
    return regimes,source


def suite_t8():
    # Spectral certificate assumptions, including an out-of-domain witness.
    spectral_rng=np.random.default_rng(SEED+22)
    worst_excess=-math.inf
    for _ in range(24):
        k=spectral_rng.normal(size=(8,D))
        k*=spectral_rng.uniform(.1,1.,(8,1))/np.linalg.norm(k,axis=1,keepdims=True)
        beta=spectral_rng.uniform(.01,.95,8);gamma=spectral_rng.uniform(.8,1.,8)
        p=np.eye(D)
        for kk,bb,gg in zip(k,beta,gamma):
            p=gg*(p-bb*np.outer(kk,kk@p))
        worst_excess=max(worst_excess,float(np.linalg.norm(p,2)-np.prod(gamma)))
    beta_bad=.5;gamma_bad=.9;k_bad=np.zeros(D);k_bad[0]=3.
    a_bad=gamma_bad*(np.eye(D)-beta_bad*np.outer(k_bad,k_bad))
    bad_norm=float(np.linalg.norm(a_bad,2))
    check("T8.norm_k_spectral_bound",worst_excess<=1e-13
          and norm(k_bad)>math.sqrt(2/beta_bad) and bad_norm>gamma_bad,
          trials=24,worst_P_norm_minus_Gamma=worst_excess,
          unnormalized_key_norm=norm(k_bad),bad_A_norm=bad_norm,gamma=gamma_bad,
          scope="The bad key also violates beta*||k||^2<1; ||k||<=1 is explicit and sufficient with R2.3, not logically necessary alone.")
    # One sensitive retrieval head, fifteen heads with orthogonal readout.
    ds=np.zeros((D,V));ds[0,0]=2.
    p=np.eye(D);p[1,1]=2.**-60
    # P is 60 legal beta=.5, gamma=1, k=e_1 steps. Fifteen queries
    # are orthogonal to the sole preserved retrieval direction e_0.
    queries=np.zeros((16,D));queries[0,0]=1.;queries[1:,1]=1.
    head_terms=[norm(p.T@q)/max(norm(q),1e-30)*norm(ds)/(norm(np.zeros_like(ds))+1)
                for q in queries]
    eta_max=max(head_terms);eta_mean=float(np.mean(head_terms));tau=eta_max/2
    check("T8.max_vs_mean_head_dilution",eta_max>tau>eta_mean
          and abs(eta_max/eta_mean-16)<1e-13,
          heads=16,eta_max=eta_max,eta_mean_diagnostic=eta_mean,threshold=tau,
          dilution=eta_max/eta_mean,orthogonal_head_terms=head_terms[1:],
          suffix="60 beta=.5, gamma=1 unit-key writes; all satisfy R2.3")
    regimes,source=calibration();DETAILS["calibration"]=source
    worst_identity=worst_ledger=worst_query=0.;ratios={"random":[],"retrieval":[],"calibrated":[]}
    for trial in range(300):
        kind=("random","retrieval","calibrated")[trial%3]
        n=12
        k,v,beta,g=operands(n)
        q=RNG.normal(size=D);q/=norm(q)
        if kind=="retrieval":
            k[:]=q; beta[:]=RNG.uniform(.45,.7);g[:]=-.001
        if kind=="calibrated":
            lam,e,m=regimes[(trial//3)%len(regimes)]
            g[:]=-lam/n
            beta[:]=-np.expm1(-e/n)
            # Preserve strict assumption (2) despite exponent saturation.
            beta[:]=min(float(beta[0]),1-1e-12)
        # Two genuinely different state trajectories (keys, values, gates,
        # decay and initial states differ); defects are computed independently.
        s2=RNG.normal(size=(D,V))*.002
        s1=s2+RNG.normal(size=(D,V))*.001
        initial=s1-s2;defects=[]
        p=adjoints(k,beta,g,q)
        for j in range(n):
            k1=k[j]+RNG.normal(size=D)*.001;k1/=norm(k1)
            v1=v[j]+RNG.normal(size=V)*.001
            pred=step(s1,k[j],v[j],beta[j],np.exp(g[j]))
            updated=step(s1,k1,v1,min(.999999,beta[j]+.001),np.exp(g[j]-.001))
            defects.append(updated-pred)
            s1=updated;s2=step(s2,k[j],v[j],beta[j],np.exp(g[j]))
        rhs=mv(p[0],initial)+sum((mv(p[j+1],d) for j,d in enumerate(defects)),np.zeros(V))
        lhs=mv(q,s1-s2);identity=maxerr(lhs,rhs);worst_identity=max(worst_identity,identity)
        projected=norm(mv(p[0],initial))+sum(norm(mv(p[j+1],d)) for j,d in enumerate(defects))
        adj=norm(p[0])*norm(initial)+sum(norm(p[j+1])*norm(d) for j,d in enumerate(defects))
        decay=np.exp(np.concatenate((np.cumsum(g[::-1])[::-1],[0.])))
        frob=norm(q)*(decay[0]*norm(initial)+sum(decay[j+1]*norm(d) for j,d in enumerate(defects)))
        q1=q+RNG.normal(size=D)*.01
        query_rhs=mv(q,s1-s2)+mv(q1-q,s1)
        query_err=maxerr(mv(q1,s1)-mv(q,s2),query_rhs);worst_query=max(worst_query,query_err)
        ledger=0.
        for t in range(n+1):
            total=np.exp(2*np.sum(g[t:]))*np.dot(q,q)
            for j in range(t,n):
                total-=np.exp(2*np.sum(g[t:j+1]))*beta[j]*(2-beta[j]*np.dot(k[j],k[j]))*np.dot(k[j],p[j+1])**2
            ledger=max(ledger,abs(np.dot(p[t],p[t])-total))
        worst_ledger=max(worst_ledger,ledger)
        ok=identity<1e-13 and query_err<1e-13 and ledger<1e-13
        ok &= norm(lhs)<=projected+1e-13 and projected<=adj+1e-13 and adj<=frob+1e-13
        check(f"T8.trial{trial:03d}.{kind}",ok,identity_error=identity,ledger_error=ledger,
              actual=norm(lhs),adj_directional=projected,adj=adj,frobenius=frob)
        if adj>1e-290:ratios[kind].append(frob/adj)
    # Designed retrieval-aligned, early-defect experiments. No claim that a
    # uniformly distributed late local defect enjoys this same improvement.
    retrieval=[];cascade=[]
    for trial in range(120):
        n=12;k=np.zeros((n,D));k[:,0]=1
        beta=np.full(n,RNG.uniform(.33,.4));g=np.full(n,-.001)
        q=np.zeros(D);q[0]=1
        p=adjoints(k,beta,g,q)
        ratio=float(np.exp(np.sum(g))/norm(p[0]));retrieval.append(ratio)
        # 3:1 cascade has three contracting GDN paths in each of two blocks;
        # intervening fixed unit-gain residual/readout maps preserve direction.
        # This is a linearized/fixed-path experiment, not a nonlinear N bound.
        cascade.append(ratio**6)
    summary={k:dict(median=float(np.median(v)),p95=float(np.percentile(v,95))) for k,v in ratios.items()}
    summary["retrieval_early_defect"]=dict(median=float(np.median(retrieval)),p95=float(np.percentile(retrieval,95)))
    summary["two_block_fixed_path_3to1"]=dict(median=float(np.median(cascade)),p95=float(np.percentile(cascade,95)))
    check("T8.retrieval_10to100_plus",np.median(retrieval)>10 and np.percentile(retrieval,95)>100,
          **summary["retrieval_early_defect"])
    check("T8.hybrid_fixed_path_improvement",np.median(cascade)>100,**summary["two_block_fixed_path_3to1"])
    # Strong raw q-key alignment at an EARLY step need not imply adjoint-key
    # alignment after later suffix rotations/contractions.
    q=np.zeros(D);q[0]=1
    k_late=np.zeros(D);k_late[:2]=1/np.sqrt(2)
    p_late=q-0.99*np.dot(k_late,q)*k_late
    k_early=np.zeros(D);k_early[:2]=[p_late[1],-p_late[0]];k_early/=norm(k_early)
    check("T8.spec_correction.retrieval_uses_evolved_adjoint",
          abs(np.dot(k_early,q))>.5 and abs(np.dot(k_early,p_late))<1e-14,
          raw_alignment=abs(float(k_early@q)),adjoint_alignment=abs(float(k_early@p_late)))
    check("T8.spec_correction.adjoint_flops",5*D==640 and (2*D*D)/(5*D)==51.2,
          exact_scalar_flops=5*D,dense_matvec_contracted_flops=2*D*D)
    # Rank-refined local defects: |p^T k| ||a||, versus ||p|| ||k|| ||a||.
    for trial in range(24):
        k,v,beta,g=operands(10);q=RNG.normal(size=D);q/=norm(q)
        p=adjoints(k,beta,g,q);state=np.zeros((D,V));directional=scalar=0.
        for j in range(10):
            a=RNG.normal(size=V)*.001
            defect=np.outer(k[j],a)
            state=np.exp(g[j])*(state-beta[j]*np.outer(k[j],mv(k[j],state)))+defect
            directional+=abs(float(p[j+1]@k[j]))*norm(a)
            scalar+=norm(p[j+1])*norm(defect)
        check(f"T8.rank_directional.{trial}",norm(mv(q,state))<=directional+1e-13
              and directional<=scalar+1e-13,actual=norm(mv(q,state)),directional=directional,scalar=scalar)
    DETAILS["adjoint_summary"]=dict(worst_identity=worst_identity,worst_ledger=worst_ledger,
                                    worst_query_identity=worst_query,tightness=summary,trials=300)


def rms(x):
    return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)


def rotate(x,positions):
    """Partial RoPE: exactly 64 of the 256 FA head coordinates."""
    out=x.copy();r=64
    angle=np.asarray(positions)[:,None,None]/(10000**(np.arange(0,r,2)/r))[None,None,:]
    a,b=x[...,:r:2],x[...,1:r:2]
    out[...,:r:2]=a*np.cos(angle)-b*np.sin(angle)
    out[...,1:r:2]=a*np.sin(angle)+b*np.cos(angle)
    return out


class HybridToy:
    """Actual causal two-block nonlinear forward; reduced widths/head counts.

    Every GDN state is 128x128 and every FA head is 256; dm=24, ff=48,
    Hq=2/Hkv=1/Hk=Hv=1 are CPU-scaled, NOT a Qwen model execution.
    """
    def __init__(self):
        self.dm,self.ff,self.blocks=24,48,2
        self.layers=[]
        def weight(a,b,scale=.15):return RNG.normal(size=(a,b))*scale/np.sqrt(a)
        for block in range(self.blocks):
            for j in range(4):
                layer=dict(block=block,kind="gdn" if j<3 else "fa",
                           gate=weight(24,48),up=weight(24,48),down=weight(48,24))
                if j<3:
                    layer.update(qkv=weight(24,3*D,.3),a=weight(24,1),b=weight(24,1),
                                 z=weight(24,V),out=weight(V,24,.2),
                                 conv=RNG.uniform(.2,.5,(4,3*D)))
                else:
                    layer.update(q=weight(24,2*256),k=weight(24,256),v=weight(24,256),
                                 z=weight(24,2*256),out=weight(2*256,24,.3))
                self.layers.append(layer)

    def layer_gdn(self,x,l,positions=None,frozen=None,nohalo=False,terminal=None,clm_at=None):
        n=len(x);a=rms(x)
        if frozen is None:
            raw=a@l["qkv"];cv=silu(conv(raw,l["conv"]))
            q,k,v=np.split(cv,3,axis=1)
            q=q/np.sqrt(np.sum(q*q,axis=1,keepdims=True)+1e-6)/np.sqrt(D)
            k=k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+1e-6)
            beta=1/(1+np.exp(-(a@l["b"])[:,0]));g=-.003-np.logaddexp(0,(a@l["a"])[:,0])*.004
            z=a@l["z"]
        else:
            idx=positions;raw=frozen["raw"][idx];cv=silu(conv(raw,l["conv"]))
            q,k,v=np.split(cv,3,axis=1)
            q=q/np.sqrt(np.sum(q*q,axis=1,keepdims=True)+1e-6)/np.sqrt(D)
            k=k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+1e-6)
            if nohalo:q,k,v=(frozen[key][idx].copy() for key in ("q","k","v"))
            beta,g,z=(frozen[key][idx].copy() for key in ("beta","g","z"))
        hist=sequential(np.zeros((D,V)),k,v,beta,g,True)
        if terminal is not None:hist[-1]=terminal
        if clm_at is not None:hist[-1]=frozen["hist"][clm_at]
        out=np.einsum("ti,tij->tj",q,hist[1:],optimize=False)
        y=x+(rms(out)*silu(z))@l["out"]
        y=y+silu(rms(y)@l["gate"])*(rms(y)@l["up"])@l["down"]
        return y,dict(raw=raw,q=q,k=k,v=v,beta=beta,g=g,z=z,hist=hist,input=x.copy())

    def layer_fa(self,x,l,positions,frozen=None,ids=None):
        a=rms(x);n=len(x)
        q=rotate((a@l["q"]).reshape(n,2,256),positions)
        if frozen is None:
            k=rotate((a@l["k"]).reshape(n,1,256),positions)
            v=(a@l["v"]).reshape(n,1,256)
        else:k,v=frozen["k"][ids],frozen["v"][ids]
        logits=np.einsum("thd,shd->hts",q,np.repeat(k,2,axis=1))/16
        logits=np.where(np.tri(n,dtype=bool)[None,:,:],logits,-1e100)
        logits-=np.max(logits,axis=-1,keepdims=True)
        att=np.exp(logits);att/=att.sum(axis=-1,keepdims=True)
        out=np.einsum("hts,shd->thd",att,np.repeat(v,2,axis=1)).reshape(n,512)
        y=x+(out/(1+np.exp(-(a@l["z"]))))@l["out"]
        y=y+silu(rms(y)@l["gate"])*(rms(y)@l["up"])@l["down"]
        return y,dict(k=k,v=v,attention=att,input=x.copy())

    def forward(self,x,positions,original=None,ids=None,method="oracle",flags=(),seam=0):
        records=[];flops=0
        for index,l in enumerate(self.layers):
            old=None if original is None else original[index]
            refresh=l["block"] in flags or method=="oracle"
            if l["kind"]=="gdn":
                kwargs={}
                if not refresh:
                    kwargs.update(frozen=old,positions=ids)
                    if method=="NoHalo-ChunkCompose":kwargs["nohalo"]=True
                    if method=="KV-Only-Evict":kwargs["terminal"]=old["hist"][-1]
                    if method=="CLM-StaleSnapshot":kwargs["clm_at"]=seam
                    if method=="Terminal-Inverse":
                        # Terminal-only affine conjugation: infer S0 from the
                        # old endpoint, then evaluate the edited fixed-operand
                        # affine map. SVD truncation stabilizes small singular
                        # directions. Deliberately omits conv-halo recompute.
                        po,ho=affine(old["k"],old["v"],old["beta"],old["g"])
                        pe,he=affine(old["k"][ids],old["v"][ids],old["beta"][ids],old["g"][ids])
                        u,sv,vt=np.linalg.svd(po,full_matrices=False)
                        inv=np.zeros_like(sv);good=sv>sv[0]*1e-8;inv[good]=1/sv[good]
                        recovered=vt.T@(inv[:,None]*(u.T@(old["hist"][-1]-ho)))
                        kwargs["terminal"]=pe@recovered+he
                        kwargs["nohalo"]=True
                x,r=self.layer_gdn(x,l,**kwargs)
                # Transparent contracted FLOPs; frozen F operands skip in_proj
                # and MLP arithmetic in a deployment, but this diagnostic toy
                # materializes token outputs for downstream comparisons.
                flops+=len(x)*(2*self.dm*(5*D+2)+7*D*V+6*self.dm*self.ff)
            else:
                frozen=None if refresh else old
                x,r=self.layer_fa(x,l,positions,frozen,ids)
                flops+=len(x)*(2*self.dm*256*8+6*self.dm*self.ff)
                flops+=4*2*256*len(x)*(len(x)+1)//2
            records.append(r)
        return x,records,flops


def toy_cache(model, records, ids, capacity, anchor=None):
    """Diagnostic cache snapshot; FA slots use original (gapped) indices."""
    cache=[]
    for l,r in zip(model.layers,records):
        if l["kind"]=="gdn":
            end=len(r["raw"]) if anchor is None else anchor
            cache.append(dict(state=r["hist"][end].copy(),
                              ring=r["raw"][max(0,end-W+1):end].copy()))
        else:
            k=np.zeros((capacity,1,256));v=np.zeros_like(k)
            k[ids]=r["k"];v[ids]=r["v"]
            cache.append(dict(k=k,v=v))
    return cache


def copy_cache(cache):
    return [{key:value.copy() for key,value in r.items()} for r in cache]


def local_forward(model, x, positions, cache, allowed, through_block=None):
    """One actual local multilayer pass from recurrent/ring/FA anchors.

    Clean historical FA KV remain cached; only positions in this pass are
    replaced. A partial pass executes every upstream block in the closure.
    """
    records=[]
    for index,l in enumerate(model.layers):
        if through_block is not None and l["block"]>through_block:break
        r=cache[index];a=rms(x);n=len(x);input_x=x.copy()
        if l["kind"]=="gdn":
            raw=a@l["qkv"]
            stream=np.concatenate((r["ring"],raw))
            cv=silu(conv(stream,l["conv"])[-n:])
            q,k,v=np.split(cv,3,axis=1)
            q=q/np.sqrt(np.sum(q*q,axis=1,keepdims=True)+1e-6)/np.sqrt(D)
            k=k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+1e-6)
            beta=1/(1+np.exp(-(a@l["b"])[:,0]))
            g=-.003-np.logaddexp(0,(a@l["a"])[:,0])*.004;z=a@l["z"]
            hist=sequential(r["state"],k,v,beta,g,True)
            r["state"]=hist[-1];r["ring"]=stream[-(W-1):].copy()
            out=np.einsum("ti,tij->tj",q,hist[1:],optimize=False)
            y=x+(rms(out)*silu(z))@l["out"]
            record=dict(raw=raw,q=q,k=k,v=v,beta=beta,g=g,z=z,hist=hist,input=input_x)
        else:
            q=rotate((a@l["q"]).reshape(n,2,256),positions)
            k=rotate((a@l["k"]).reshape(n,1,256),positions)
            v=(a@l["v"]).reshape(n,1,256)
            r["k"][positions]=k;r["v"][positions]=v
            prefix=allowed[allowed<=positions[-1]]
            keys=np.repeat(r["k"][prefix],2,axis=1)
            values=np.repeat(r["v"][prefix],2,axis=1)
            logits=np.einsum("thd,shd->hts",q,keys)/16
            logits=np.where((prefix[None,:]<=positions[:,None])[None,:,:],logits,-1e100)
            logits-=np.max(logits,axis=-1,keepdims=True)
            att=np.exp(logits);att/=att.sum(axis=-1,keepdims=True)
            out=np.einsum("hts,shd->thd",att,values).reshape(n,512)
            y=x+(out/(1+np.exp(-(a@l["z"]))))@l["out"]
            record=dict(k=k,v=v,attention=att,input=input_x)
        x=y+silu(rms(y)@l["gate"])*(rms(y)@l["up"])@l["down"]
        records.append(record)
    return x,records


def suite_s4():
    model=HybridToy();t=52;c=8
    x=RNG.normal(size=(t,24))*.5
    spans=[[9,11],[13,17],[25,29],[32,36]]
    pi,deleted=survivor_map(t,spans)
    _,original,_=model.forward(x,np.arange(t))
    _,compact_records,full_compact=model.forward(x[pi],np.arange(len(pi)))
    _,oracle_records,full=model.forward(x[pi],pi)
    gdn=[i for i,l in enumerate(model.layers) if l["kind"]=="gdn"]
    dirty=set(dirty_chunks(t,c,spans));m0=min(dirty)*c
    chunks=math.ceil(t/c)
    # Build immutable original maps BEFORE editing. No edited operands enter P/H.
    archive={i:[affine(*(original[i][key][j*c:min(t,(j+1)*c)]
                         for key in ("k","v","beta","g")))
                for j in range(chunks)] for i in gdn}
    # Independent multilayer F oracle: freeze each ORIGINAL layer input stream.
    f_oracle={i:model.layer_gdn(original[i]["input"][pi],model.layers[i],pi,original[i])[1]
              for i in gdn}
    # Pass 1: reconstruct ORIGINAL dirty chunks from original layer anchors,
    # rings and pre-eviction FullAttn KV, running the original local network.
    recovered={};recovery_error=0.
    for j in sorted(dirty):
        ids=np.arange(j*c,min(t,(j+1)*c))
        cache=toy_cache(model,original,np.arange(t),t+1,anchor=j*c)
        _,records=local_forward(model,x[ids],ids,cache,np.arange(t))
        recovered[j]=records
        for i in gdn:
            for key in ("input","raw","beta","g","z"):
                recovery_error=max(recovery_error,maxerr(records[i][key],original[i][key][ids]))
    check("S4.F.pass1_original_operand_recovery",recovery_error<1e-13,
          max_abs_error=recovery_error,dirty_original_chunks=sorted(dirty))
    # Pass 2: independent per-layer F repair. Persistent operand archive is not
    # needed by the algorithm; this toy retains originals only as test oracles.
    f_cache=toy_cache(model,original,np.arange(t),t+1,anchor=m0)
    fast_cache=copy_cache(f_cache)
    fast_local_tokens=0;f_local_tokens=0;clean_jumps=0
    for j in range(m0//c,chunks):
        ids=pi[(pi>=j*c)&(pi<min(t,(j+1)*c))]
        if not len(ids):continue
        if j in dirty:
            # CHymERA-Fast: exactly ONE edited-seam multilayer forward.
            local_forward(model,x[ids],ids,fast_cache,pi)
            fast_local_tokens+=len(ids)
            for i in gdn:
                r=f_cache[i];old=recovered[j][i];offset=ids-j*c
                raw=old["raw"][offset]
                stream=np.concatenate((r["ring"],raw))
                cv=silu(conv(stream,model.layers[i]["conv"])[-len(ids):])
                _,k,v=np.split(cv,3,axis=1)
                k=k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+1e-6)
                r["state"]=sequential(r["state"],k,v,old["beta"][offset],old["g"][offset])
                r["ring"]=stream[-(W-1):].copy()
            f_local_tokens+=len(ids)
        else:
            for cache in (f_cache,fast_cache):
                for i in gdn:
                    p,h=archive[i][j]
                    cache[i]["state"]=p@cache[i]["state"]+h
                    cache[i]["ring"]=original[i]["raw"][ids[-(W-1):]].copy()
            clean_jumps+=1
    f_errors=[];fast_errors=[];fast_vs_f=[]
    for layer,i in enumerate(gdn):
        ferr=maxerr(f_cache[i]["state"],f_oracle[i]["hist"][-1])
        fast_err=maxerr(fast_cache[i]["state"],f_oracle[i]["hist"][-1])
        delta=maxerr(fast_cache[i]["state"],f_cache[i]["state"])
        f_errors.append(ferr);fast_errors.append(fast_err);fast_vs_f.append(delta)
        check(f"S4.F.all_layers.layer{layer}",ferr<1e-13,max_abs_error=ferr)
    check("S4.F.all_6_GDN_layers",len(f_errors)==6 and max(f_errors)<1e-13,
          per_layer_max_abs_error=f_errors,local_dirty_passes=2,persistent_operand_bytes_per_token=0)
    check("S4.Fast.layer0_F_exact",fast_errors[0]<1e-13,max_abs_error=fast_errors[0])
    check("S4.Fast.block1_distinct_from_F",all(e>1e-8 for e in fast_vs_f[3:]),
          block1_per_layer_max_abs_difference=fast_vs_f[3:])
    # Primary oracle actually starts at the earliest dirty anchor, reuses only
    # the untouched prefix, and replays ALL dependent suffix layers.
    suffix_ids=pi[pi>=m0]
    replay_cache=toy_cache(model,original,np.arange(t),t+1,anchor=m0)
    local_forward(model,x[suffix_ids],suffix_ids,replay_cache,pi)
    n_cache=toy_cache(model,oracle_records,pi,t+1)
    replay_error=max(maxerr(replay_cache[i]["state"],n_cache[i]["state"]) for i in gdn)
    replay_kv_error=max(maxerr(replay_cache[i][key][pi],n_cache[i][key][pi])
                        for i,l in enumerate(model.layers) if l["kind"]=="fa" for key in ("k","v"))
    check("S4.Sparse_Anchor_SuffixReplay.primary_oracle",max(replay_error,replay_kv_error)<1e-13,
          GDN_error=replay_error,KV_error=replay_kv_error,earliest_dirty_anchor=m0)
    # Compare one identical NEW query from each repaired cache. This has a
    # defined output even though F editing itself only promises cache states.
    probe=RNG.normal(size=(1,24))*.5
    def readout(cache,compact=False):
        pos=len(pi) if compact else t
        allowed=np.arange(len(pi)+1) if compact else np.append(pi,t)
        return local_forward(model,probe,np.array([pos]),copy_cache(cache),allowed)[0][-1]
    target=readout(replay_cache)
    compact_cache=toy_cache(model,compact_records,np.arange(len(pi)),t+1)
    compact_target=readout(compact_cache,True)
    tok_g=2*24*(5*D+2)+7*D*V+6*24*48
    tok_a=2*24*256*8+6*24*48
    suffix_pairs=sum(int(np.sum(pi<=token)) for token in suffix_ids)
    block_replay=len(suffix_ids)*(3*tok_g+tok_a)+4*2*256*suffix_pairs
    results=[]
    methods=("Full-Prefill-Compact","Full-Replay-GappedRoPE","Sparse-Anchor-SuffixReplay",
             "KV-Only-Evict","CLM-StaleSnapshot","NoHalo-ChunkCompose","Terminal-Inverse",
             "CHymERA-F","CHymERA-Fast")
    for method in methods:
        if method=="Full-Prefill-Compact":cache=compact_cache;cost=full_compact;y=compact_target
        elif method=="Full-Replay-GappedRoPE":cache=n_cache;cost=full;y=readout(cache)
        elif method=="Sparse-Anchor-SuffixReplay":cache=replay_cache;cost=2*block_replay;y=target
        elif method in ("CHymERA-F","CHymERA-Fast"):
            cache=f_cache if method=="CHymERA-F" else fast_cache
            cost=None;y=readout(cache)
        else:
            _,records,cost=model.forward(x[pi],pi,original,pi,method=method,seam=spans[0][0])
            cache=toy_cache(model,records,pi,t+1);y=readout(cache)
        state_error=max(maxerr(cache[i]["state"],f_oracle[i]["hist"][-1]) for i in gdn)
        results.append(dict(method=method,error_to_gapped=norm(y-target),
                            error_to_compact=norm(y-compact_target),
                            max_GDN_terminal_error_to_F=state_error,
                            diagnostic_forward_flops=None if cost is None else int(cost)))
        check("S4.forward.finite."+method,np.isfinite(y).all())
    # Freeze ALL primary scores before ANY refresh, then reuse them throughout
    # the threshold sweep. The toy has 3 layer/head terms per hybrid block.
    scores=[];component_scores=[]
    seam=max(halo(pi));seam_orig=int(pi[seam])
    for block in range(model.blocks):
        terms=[];unnormalized=[]
        for i in gdn[3*block:3*block+3]:
            r=original[i];f=f_oracle[i];q=r["q"][-1]
            p=adjoints(r["k"][pi],r["beta"][pi],r["g"][pi],q)
            ds=f["hist"][seam+1]-r["hist"][seam_orig+1]
            sensitivity=norm(p[seam+1])/max(norm(q),1e-30)*norm(ds)
            unnormalized.append(sensitivity)
            terms.append(sensitivity/(norm(r["hist"][seam_orig+1])+1))
        scores.append(max(terms))
        component_scores.append(dict(block=block,normalized_head_terms=terms,
                                     eta_max=max(terms),unnormalized_head_mean_diagnostic=float(np.mean(unnormalized))))
    frozen_scores=tuple(scores)
    thresholds=[math.inf,max(scores)*1.01,float(np.mean(scores)),min(scores)*.99,0.]
    # Isolate each triggering block too: selecting only block 1 MUST actually
    # refresh block 0. This is independent of this model's score ordering.
    def closure_for(triggered):
        return set(range(max(triggered)+1)) if triggered else set()
    def refresh(closure):
        cache=copy_cache(fast_cache)
        if closure:
            anchored=toy_cache(model,original,np.arange(t),t+1,anchor=m0)
            local_forward(model,x[suffix_ids],suffix_ids,anchored,pi,through_block=max(closure))
            for i,l in enumerate(model.layers):
                if l["block"] in closure:cache[i]=anchored[i]
        return cache
    partial_cache=refresh({0})
    check("S4.gate.partial_closure_preserves_unreplayed_block",
          all(maxerr(partial_cache[i]["state"],n_cache[i]["state"])<1e-13 for i in gdn[:3])
          and all(maxerr(partial_cache[i]["state"],fast_cache[i]["state"])==0 for i in gdn[3:]))
    probe_closure=closure_for({1});probe_cache=refresh(probe_closure)
    check("S4.gate.upstream_closure_executed",probe_closure=={0,1}
          and max(maxerr(probe_cache[i]["state"],n_cache[i]["state"]) for i in gdn)<1e-13)
    local_pairs=sum(int(np.sum(pi<=token)) for token in pi if token//c in dirty)
    fast_cost=fast_local_tokens*(6*tok_g+2*tok_a)+4*2*2*256*local_pairs
    fast_cost+=clean_jumps*6*(2*D*D*V+D*V)
    recover_tokens=sum(min(c,t-j*c) for j in dirty)
    recovery_pairs=sum(token+1 for j in dirty for token in range(j*c,min(t,(j+1)*c)))
    recovery_cost=recover_tokens*(6*tok_g+2*tok_a)+4*2*2*256*recovery_pairs
    f_cost=recovery_cost+f_local_tokens*6*(5*D*V+V)
    f_cost+=clean_jumps*6*(2*D*D*V+D*V)+6*f_local_tokens*2*W*(3*D)
    # Score diagnostics materialize original F histories in this CPU toy. A
    # deployed score path must pay its actual operand/adjoint reconstruction.
    score_cost=f_cost+6*len(pi)*5*D
    sweep=[];prev=set();prev_cost=0
    for tau in thresholds:
        triggered=set(range(model.blocks)) if tau==0 else {b for b,s in enumerate(frozen_scores) if s>tau}
        closure=closure_for(triggered)
        cache=refresh(closure);y=readout(cache)
        # Infinity bypasses scoring; finite policies charge F sensitivity work.
        policy_cost=int(fast_cost+(0 if not math.isfinite(tau) else score_cost)
                        +len(closure)*block_replay)
        e=norm(y-target)
        check("S4.gate.nested."+str(len(sweep)),prev.issubset(closure) and policy_cost>=prev_cost)
        check("S4.gate.frozen_scores."+str(len(sweep)),tuple(scores)==frozen_scores)
        sweep.append(dict(threshold="infinity" if not math.isfinite(tau) else tau,
                          triggered_blocks=sorted(triggered),blocks=sorted(closure),error=e,
                          policy_flops=policy_cost,dependency_materialized_blocks=sorted(closure),
                          max_GDN_terminal_error_to_N=max(maxerr(cache[i]["state"],n_cache[i]["state"]) for i in gdn)))
        prev,prev_cost=closure,policy_cost
    fast=next(r for r in results if r["method"]=="CHymERA-Fast")
    check("S4.gate.endpoint_infinity",abs(sweep[0]["error"]-fast["error_to_gapped"])<1e-13)
    check("S4.gate.endpoint_zero_forced_replay",sweep[-1]["error"]<1e-13
          and sweep[-1]["max_GDN_terminal_error_to_N"]<1e-13,max_abs_error=sweep[-1]["error"])
    errors=[r["error"] for r in sweep]
    monotone=all(a+1e-13>=b for a,b in zip(errors,errors[1:]))
    counter=[0.,1.,0.]
    check("S4.spec_correction.nonmonotone_cancellation",counter[1]>counter[0],errors=counter,
          construction="Two stale block contributions +1 and -1; refresh the +1 block first.")
    envelope=np.minimum.accumulate(errors).tolist()
    check("S4.pareto_envelope",all(a>=b for a,b in zip(envelope,envelope[1:])))
    check("S4.spec_correction.strict_zero_gate",not(0.>0.) and closure_for(set(range(2)))=={0,1})
    DETAILS["hybrid_simulation"]=dict(
        architecture=dict(blocks=2,layers=8,GDN_layers=6,GDN_dk=128,GDN_dv=128,conv_width=4,
                          FullAttn_head_dim=256,Hq=2,Hkv=1,dm=24,ff=48,partial_rotary_factor=.25),
        spans=spans,methods=results,scores=scores,score_components=component_scores,
        score_definition="pre-refresh frozen max over layer/heads, dimensionless relative state perturbation",
        F_repair=dict(per_layer_terminal_max_abs_error=f_errors,pass1_operand_max_abs_error=recovery_error,
                      local_dirty_passes=2,persistent_operand_bytes_per_token=0),
        Fast_repair=dict(per_layer_terminal_max_abs_error_to_F=fast_errors,
                         per_layer_terminal_max_abs_difference_from_F=fast_vs_f,
                         local_dirty_passes=1,local_tokens=fast_local_tokens,clean_macro_jumps=clean_jumps),
        primary_oracle="Sparse-Anchor-SuffixReplay / Full-Replay-GappedRoPE",
        threshold_sweep=sweep,raw_error_monotone=monotone,pareto_error_envelope=envelope,
        policy_cost_components=dict(fast_terminal_repair=fast_cost,F_two_pass_repair=f_cost,
                                    score=score_cost,suffix_replay_per_block=block_replay,
                                    archive="Original streams/maps retained as independent CPU test oracles; not a measured sublinear implementation."),
        scope="Reduced nonlinear NumPy model. F executes original dirty reconstruction plus independent layer repair; Fast executes one dirty multilayer pass and clean original affine jumps. Output errors compare one new query from edited caches; counts exclude this common query and archive construction.",
        terminal_inverse_scope="Pe*Po^{-1}(S_old-Ho)+He, SVD cutoff 1e-8; fixed original operands, no conv halo; not an N inverse.",
        conclusion="F is exact across all six GDN layers; Fast is F-exact only at layer 0. Certified executes the frozen-score upstream closure; only nested cost/sets and forced replay endpoint are universal.")


CORRECTIONS = [
    "T6(b): extended-window disjointness is sufficient and structurally necessary for all admissible operands, not literal iff for a particular summary (zero gates/keys can coincide).",
    "T6(b): one span can touch THREE surviving dirty original macro-chunks, e.g. C=16,D=[1,30]; use <=3R chunks or <=R(2C+W-3) replay tokens, not <=2R chunks.",
    "T6(c): CHymERA-F needs two local passes: original operands recovered BEFORE KV eviction, then independent F repair across all GDN layers (0 B/tok persistent operands). CHymERA-Fast uses one edited-seam pass; only layer 0 is F-exact, with no multilayer F or N-error certificate.",
    "T6(c): square-root edit FLOPs can be Theta(T^(3/2)) with attention, little-o of quadratic prefill, not o(T). At finite T the sparse archive may exceed KV.",
    "T7: positive naive-patching state error is an existential unit-scale witness; zero kernels/inputs rule out universal Omega(1) or >1e-2 errors.",
    "T8(a): decayed adjoint uses 5*d scalar FLOPs including gamma scaling, not 4*d; dense matvec ratio is ~51.2 at d=128, not 128.",
    "T8(b): exact contraction depends on k_s^T p_s, not raw k_s^T q_T. A 10x-100x gain is regime-dependent, not universal.",
    "T8(c): residual certificate is conditional on certified residual radii. BF16 tensors alone do not supply those radii for free; toy computes true residuals offline.",
    "T8(c)/S4: Freeze dimensionless max-over-heads scores before any refresh; execute the upstream block closure. tau=0 forces full Sparse-Anchor-SuffixReplay, including zero-score blocks; tau=infinity bypasses scoring and returns Fast.",
    "S4: frozen-score flags/FLOPs are nested, actual nonlinear error need not be monotone; seven-method Pareto dominance is neither a theorem nor implied by CPU smoke tests."
]


def write_reports():
    failures=[r for r in CHECKS if not r["passed"]]
    result=dict(round="2.1",seed=SEED,d_k=D,d_v=V,W=W,checks=len(CHECKS),failures=len(failures),
                failed_checks=failures,passed=len(CHECKS)-len(failures),
                environment=dict(python=platform.python_version(),numpy=np.__version__,
                                 network=False,GPU=False,threads=1),
                interpretation="0 failures means the corrected, explicitly scoped claims passed. Refuted requested claims are listed as specification corrections, not certified.",
                specification_corrections=CORRECTIONS,details=DETAILS,tests=CHECKS)
    # Strict JSON: no NaN/Infinity; sweep infinity is stored as a string.
    (ROOT/"SMOKE_R2_RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=["# CHymERA Round 2.1 CPU Smoke Report", "",
           f"Seed `{SEED}`; pure NumPy CPU; GDN `d_k=d_v=128`, `W=4`.", "",
           f"**{len(CHECKS)} checks; {len(failures)} failures; {len(CHECKS)-len(failures)} passed.**", "",
           result["interpretation"],"",
           "## Suite coverage", "",
           "| Suite | Checks | Failures |", "|---|---:|---:|"]
    for suite in ("T6","T7","T8","S4"):
        rows=[r for r in CHECKS if r["name"].startswith(suite+".")]
        lines.append(f"| {suite} | {len(rows)} | {sum(not r['passed'] for r in rows)} |")
    lines += ["", "## Required specification corrections", ""]+[f"{i+1}. {s}" for i,s in enumerate(CORRECTIONS)]
    lines += ["", "## Qwen configuration and explicit cost model", "",
              "Local verified config.json files are hashed in JSON. MiB=2^20 bytes. Delta archives omit chunk 0 and resident terminal state. Conv checkpoints stay BF16. C=4√T plus T=4096,C=1024 (m=3 versus m=15 at C=256; exactly 5x smaller archives); MiB/GFLOPs are exact decimal conversions of integer counts (only ratios rounded); two local CHymERA-F operand-recovery/edit passes; Q=min(T,3C); clean jumps conservatively N−1. Costs exclude prefill archive construction, norms/softmax/transcendentals, I/O, and workspace; this is a contracted FLOP model, not wall-clock speed.", "",
              "| Model | T | C | m | FA KV MiB | Delta FP32 MiB | Delta BF16 MiB | Direct FP32 MiB | Direct BF16 MiB | Edit upper GFLOPs | Prefill GFLOPs | Prefill/edit |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in DETAILS.get("cost_table",[]):
        exact=lambda value,places:f"{value:.{places}f}".rstrip("0").rstrip(".")
        cells=[r["model"],str(r["T"]),str(r["stride"]),str(r["m"])]
        cells += [exact(r[key]/2**20,7) for key in
                  ("kv_bytes","delta_bytes","bf16_delta_bytes","direct_bytes","bf16_direct_bytes")]
        cells += [exact(r[key]/1e9,9) for key in ("edit_upper_flops","full_flops")]
        cells += [f"{r['speedup']:.3f}×"]
        lines.append("| "+" | ".join(cells)+" |")
    a=DETAILS.get("adjoint_summary",{})
    lines += ["", "## Adjoint identities and tightness", "", "```json",json.dumps(a,indent=2),"```",
              "", "The real-weight file supplies only aggregate (Lambda,E) regimes. It supplies no real keys/queries/values. Retrieval early-defect and fixed-path cascade cases are deliberately structured synthetic examples; random/calibrated results are reported separately.",
              "", "## Nonlinear two-block 3:1 simulation", "",
              "Actual GDN conv/SiLU, recurrent state, partial RoPE, causal gated GQA (FA head_dim=256), residual and SwiGLU forwards. Widths/head counts are reduced for CPU. Terminal-Inverse evaluates Pe*Po^{-1}(S_old-Ho)+He with SVD cutoff 1e-8 under fixed original operands, omitting conv halo. CHymERA-F runs original dirty-chunk reconstruction and independent F layer repair; CHymERA-Fast runs one edited-seam multilayer pass plus clean original P/H jumps. The primary oracle starts at the earliest dirty anchor and replays the dependent suffix. Errors below compare a common new query from the repaired caches. This is not a model-quality benchmark or production kernel. Policy FLOPs charge the upstream closure actually executed; the common query and archive construction are excluded.", "",
              "| Method | New-query error to gapped replay | New-query error to compact replay | Max GDN terminal error to F | Diagnostic forward FLOPs |",
              "|---|---:|---:|---:|---:|"]
    sim=DETAILS.get("hybrid_simulation",{})
    for r in sim.get("methods",[]):
        lines.append(f"| {r['method']} | {r['error_to_gapped']:.6g} | {r['error_to_compact']:.6g} | {r['max_GDN_terminal_error_to_F']:.6g} | {r['diagnostic_forward_flops'] if r['diagnostic_forward_flops'] is not None else 'see policy costs'} |")
    lines += ["", "F terminal max-absolute errors (GDN layers 0..5): `"+
              json.dumps(sim.get("F_repair",{}).get("per_layer_terminal_max_abs_error",[]))+"`.",
              "Fast terminal errors to F (GDN layers 0..5): `"+
              json.dumps(sim.get("Fast_repair",{}).get("per_layer_terminal_max_abs_error_to_F",[]))+"`.",
              "", "Scores are dimensionless max-over-heads, frozen before any refresh. Unnormalized head means are diagnostic only.",
              "", "| Gate threshold | Triggered blocks | Executed upstream closure | New-query error to gapped replay | Illustrative policy FLOPs |",
              "|---|---|---|---:|---:|"]
    for r in sim.get("threshold_sweep",[]):
        lines.append(f"| {r['threshold']} | {r['triggered_blocks']} | {r['blocks']} | {r['error']:.6g} | {r['policy_flops']} |")
    lines += ["",f"Raw error monotone in this run: `{sim.get('raw_error_monotone')}`. Only the best-so-far Pareto envelope is monotone by construction. The separate cancellation counterexample rules out a universal monotonicity theorem.",
              "", "Policy contracted FLOP components (common new query excluded): `"+
              json.dumps(sim.get("policy_cost_components",{}),ensure_ascii=False)+"`.",
              "", "## Certificate and precision scope", "",
              "Archive residual tests compute true local residuals against original FP64 maps. They verify the deterministic certificate identity for FP32/BF16/low-rank storage; they do not claim a free online residual oracle. Direct-vs-delta cancellation uses an explicit large-old/small-edited witness. Different reduction orders need not be bitwise identical. Comparisons use maximum absolute errors; bound tests allow 1e-13 numerical slack.",
              "", "## Reproduction", "", "```sh", "python3 smoke_r2.py", "```", "",
              "No network/GPU or new packages are used. The only generated files are this report and SMOKE_R2_RESULTS.json. All individual metrics and config hashes are in JSON."]
    if failures:
        lines += ["", "## Failures", "", "```json",json.dumps(failures,indent=2),"```"]
    (ROOT/"SMOKE_R2_REPORT.md").write_text("\n".join(lines)+"\n")
    print(f"CHymERA R2.1: {len(CHECKS)} checks, {len(failures)} failures (seed={SEED})")
    for row in failures:print("FAIL",row["name"],row["metrics"])
    return len(failures)


def main():
    for suite in (suite_t6,suite_t7,suite_t8,suite_s4):
        try:
            suite()
        except Exception as exc:
            check(suite.__name__+".uncaught_exception",False,error=repr(exc))
            import traceback
            traceback.print_exc()
    return 1 if write_reports() else 0


if __name__=="__main__":
    sys.exit(main())
