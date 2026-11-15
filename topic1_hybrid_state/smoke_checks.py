#!/usr/bin/env python3
"""Round-1 mathematical smoke checks. NumPy only; no downloads or GPU.

Seed-fixed synthetic operators, not tests of pretrained model quality.
Every theorem number maps to a test_* function. Times are observations, not asserts.
Run: python3 smoke_checks.py
"""
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

SEED = 20261109
D = V = 128
EPS = 1e-6
RNG = np.random.default_rng(SEED)
CHECKS = []
DETAILS = {}


def norm(x):
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def opnorm(x):
    return float(np.linalg.norm(x, ord=2))


def check(name, error, limit, **extra):
    error, limit = float(error), float(limit)
    ok = bool(np.isfinite(error) and error <= limit)
    CHECKS.append(dict(name=name, passed=ok, error=error, limit=limit, **extra))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {error:.5g} <= {limit:.5g}")


def close(name, x, y, tol=2e-11):
    err = norm(np.asarray(x) - np.asarray(y)) / max(1., norm(y))
    check(name, err, tol)
    return err


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def l2(x, scale=1.):
    return x / np.sqrt(np.sum(x*x, axis=-1, keepdims=True) + EPS) * scale


def rms(x):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + EPS)


def seq(k, v, beta, gamma):
    return dict(k=np.asarray(k), v=np.asarray(v), beta=np.asarray(beta), gamma=np.asarray(gamma))


def cast(p, dtype):
    return {k: x.astype(dtype) for k, x in p.items()}


def cut(p, lo, hi):
    return {k: v[lo:hi] for k, v in p.items()}


def step(S, k, v, beta, gamma):
    # Exact operation order of the reference recurrent kernel (FP32 if inputs FP32).
    decayed = S * gamma
    mem = np.sum(decayed * k[:, None], axis=0)
    delta = (v - mem) * beta
    return decayed + k[:, None] * delta[None, :]


def linear(S, k, beta, gamma):
    decayed = gamma * S
    return decayed - (beta*k)[:, None] * (k @ decayed)[None, :]


def run(p, initial=None, states=False):
    dtype = p['k'].dtype
    S = np.zeros((D, V), dtype=dtype) if initial is None else initial.astype(dtype).copy()
    hist = [S.copy()] if states else None
    for k, v, b, g in zip(p['k'], p['v'], p['beta'], p['gamma']):
        S = step(S, k, v, b, g)
        if states:
            hist.append(S.copy())
    return (S, hist) if states else S


def propagator(p):
    P = np.eye(D, dtype=p['k'].dtype)
    for k, b, g in zip(p['k'], p['beta'], p['gamma']):
        P = linear(P, k, b, g)
    return P


def inverse_step(X, k, B, beta, gamma):
    dtype = X.dtype.type
    a2 = np.sum(k*k)
    alpha = dtype(beta / (dtype(1) - beta*a2))
    zeta = dtype(dtype(1) / gamma)
    Y = X - B
    c = np.sum(k[:, None] * Y, axis=0)
    H = (alpha*k)[:, None] * c[None, :]
    return zeta * (Y + H)


def Bs(p):
    return p['beta'][:, None, None]*p['k'][:, :, None]*p['v'][:, None, :]


def test_t1():
    m, r = 12, 7
    basis, _ = np.linalg.qr(RNG.normal(size=(D, r)))
    keys = l2(RNG.normal(size=(m, r)) @ basis.T)
    p = seq(keys, RNG.normal(size=(m, V)), RNG.uniform(.1, .55, m), RNG.uniform(.97, .999, m))
    S0 = RNG.normal(size=(D, V))*.03
    P = propagator(p)
    # Equation (1), including placement of decay and orientation.
    k, b, g, v = keys[0], p['beta'][0], p['gamma'][0], p['v'][0]
    close('T1.true_kernel_order', step(S0, k, v, b, g), g*(np.eye(D)-b*np.outer(k,k)) @ S0+b*np.outer(k,v))
    close('T1.affine_core', run(p, S0), P@S0+run(p))
    gamma = float(np.prod(p['gamma']))
    D0 = RNG.normal(size=(D, V))*.02
    perp = D0-basis@(basis.T@D0)
    close('T1.orthogonal_ghost', P@perp, gamma*perp)
    X = D0.copy()
    erased = 0.
    for k, b in zip(keys, p['beta']):
        erased += b*(2-b*(k@k))*norm(k@X)**2
        X = linear(X, k, b, 1.)
    close('T1.energy_ledger', norm(X)**2, norm(D0)**2-erased)
    gain = norm(P@D0)
    check('T1.ghost_lower', gamma*norm(perp)-gain, 2e-12)
    check('T1.ghost_upper', gain-gamma*norm(D0), 2e-12)
    U, sv, Vh = np.linalg.svd(P)
    lam = -math.log(gamma)
    E = float(np.sum(-np.log1p(-p['beta']*np.sum(keys*keys,axis=1))))
    chi = -math.log(float(sv[-1]))
    check('T1.geometry_lower', lam+E/r-chi, 1e-11)
    check('T1.geometry_upper', chi-lam-E, 1e-11)
    close('T1.max_singular_perp', sv[0], gamma)
    w = RNG.normal(size=V);w /= norm(w)
    close('T1.singular_survival', P@np.outer(Vh[-1],w), sv[-1]*np.outer(U[:,-1],w))
    close('T1.singular_inverse', np.linalg.solve(P,np.outer(U[:,-1],w)), np.outer(Vh[-1],w)/sv[-1])
    m = 12
    ksame = np.repeat(np.eye(D)[0:1], m,axis=0)
    korth = np.eye(D)[:m]
    gains=[]
    for label, kk in [('aligned',ksame),('orthogonal',korth)]:
        pp=seq(kk,np.zeros((m,V)),np.full(m,.4),np.full(m,.99))
        actual=1/np.linalg.svd(propagator(pp),compute_uv=False)[-1]
        exact=.99**(-m)*.6**(-m if label=='aligned' else -1)
        close('T1.equal_logs_'+label,actual,exact)
        gains.append(float(actual))
    DETAILS['T1']=dict(r=r,m=m,Lambda=lam,E=E,chi=chi,equal_logs_gains=gains)
    zero=seq(np.zeros((3,D)),np.zeros((3,V)),np.full(3,.4),np.full(3,.99))
    close('T1.zero_key_rank0',propagator(zero),.99**3*np.eye(D))


def rounding_constant(k,b,g,u,alpha_tilde,zeta_tilde):
    a2=float(k@k);alpha=b/(1-b*a2);zeta=1/g
    eta=lambda p:p*u/(1-p*u)
    A=abs(alpha_tilde)*a2
    F=1+A*(1+eta(D))*(1+eta(2))
    cu=abs(zeta_tilde)*((1+A)*u+(1+u)*(A*(eta(D)+eta(2)*(1+eta(D)))+u*F)+u*(1+u)**2*F)
    cp=abs(zeta_tilde-zeta)*(1+abs(alpha)*a2)+abs(zeta_tilde)*abs(alpha_tilde-alpha)*a2
    return cu+cp


def test_t2():
    u=float(np.finfo(np.float32).eps/2)
    # Noncommuting inverse residual ledger, beyond the aligned stress case below.
    rg=np.random.default_rng(SEED+2)
    gp=seq(l2(rg.normal(size=(6,D))),rg.normal(size=(6,V))*.03,np.full(6,.3),np.full(6,.99))
    g0=rg.normal(size=(D,V))*.01
    end=run(gp,g0);guess=end+rg.normal(size=(D,V))*1e-8
    gs=[None]*7;gs[6]=guess
    for j in range(5,-1,-1):
        gs[j]=inverse_step(gs[j+1],gp['k'][j],Bs(gp)[j],gp['beta'][j],gp['gamma'][j])+rg.normal(size=(D,V))*1e-9
    Z=np.eye(D);ledger=np.zeros((D,V))
    for j in range(6):
        kk=gp['k'][j];bb=gp['beta'][j];gg=gp['gamma'][j]
        invA=(np.eye(D)+bb/(1-bb*(kk@kk))*np.outer(kk,kk))/gg
        Z=Z@invA
        rho=step(gs[j],kk,gp['v'][j],bb,gg)-gs[j+1]
        ledger+=Z@rho
    ledger+=Z@(guess-end)
    close('T2.noncommuting_residual_ledger',gs[0]-g0,ledger)
    # Explicit local error constant for the exact evaluation schedule in Theorem 2(b).
    local=[]
    for beta in [.1,.8,.97]:
        k=l2(RNG.normal(size=D)).astype(np.float32)
        X=RNG.normal(size=(D,V)).astype(np.float32)
        B=(RNG.normal(size=(D,V))*.1).astype(np.float32)
        b=np.float32(beta);g=np.float32(.98)
        at=np.float32(b/(np.float32(1)-b*np.sum(k*k)));zt=np.float32(1/g)
        out=inverse_step(X,k,B,b,g).astype(np.float64)
        kk=k.astype(np.float64);Y=X.astype(np.float64)-B.astype(np.float64)
        al=float(b)/(1-float(b)*(kk@kk))
        ref=(Y+al*kk[:,None]*(kk@Y)[None,:])/float(g)
        bound=rounding_constant(kk,float(b),float(g),u,float(at),float(zt))*norm(Y)
        err=norm(out-ref)
        check(f'T2.explicit_round_bound_beta{beta}',err-bound,2e-11)
        local.append(dict(beta=beta,error=err,bound=bound))
    # Natural forward and inverse FP32, oracle on identical effective operands.
    m=96;k=np.repeat(np.eye(D)[0:1],m,axis=0).astype(np.float32)
    p32=seq(k,(RNG.normal(size=(m,V))*.03).astype(np.float32),np.full(m,.2,np.float32),np.full(m,.995,np.float32))
    p64=cast(p32,np.float64);S0=RNG.normal(size=(D,V))*.02
    _,h32=run(p32,S0,states=True);_,h64=run(p64,S0,states=True)
    natural=[]
    B32=Bs(p32)
    for horizon in [1,8,16,32,64,96]:
        shat=h32[horizon].copy();back={horizon:shat.astype(np.float64)}
        for j in range(horizon-1,-1,-1):
            shat=inverse_step(shat,p32['k'][j],B32[j],p32['beta'][j],p32['gamma'][j])
            back[j]=shat.astype(np.float64)
        # Exact residual identity and certificate using diagonal Z_j for this aligned case.
        endpoint=back[horizon]-h64[horizon]
        zdiag=np.ones(D);rhs=np.zeros_like(S0);certificate=0.
        for j in range(1,horizon+1):
            kk=p64['k'][j-1];bb=p64['beta'][j-1];gg=p64['gamma'][j-1]
            rho=step(back[j-1],kk,p64['v'][j-1],bb,gg)-back[j]
            zdiag/=gg*(1-bb*kk*kk)
            rhs+=zdiag[:,None]*rho
            certificate+=float(np.max(zdiag))*norm(rho)
        rhs+=zdiag[:,None]*endpoint
        certificate+=float(np.max(zdiag))*norm(endpoint)
        actual=norm(back[0]-S0)
        close(f'T2.residual_identity_m{horizon}',back[0]-S0,rhs,1e-6)
        check(f'T2.error_certificate_m{horizon}',actual-certificate,2e-8)
        natural.append(dict(m=horizon,inverse_gain=float(max(zdiag)),rollback_absolute_error=actual,certificate=certificate,endpoint_error=norm(endpoint)))
    # Minimax boundary: inject a known endpoint uncertainty along the worst direction.
    beta=.25;gamma=.99;delta=2.**-24;tau=.01
    loggain=-math.log(gamma)-math.log1p(-beta)
    ms=math.ceil(math.log(tau/delta)/loggain)
    boundary=[]
    for n in [ms-1,ms,ms+1]:
        gain=math.exp(n*loggain);radius=delta*gain
        endpoint=np.zeros((D,V));endpoint[0,0]=delta
        pp=seq(np.repeat(np.eye(D)[0:1],n,axis=0),np.zeros((n,V)),np.full(n,beta),np.full(n,gamma))
        recover=np.linalg.solve(propagator(pp),endpoint)
        close(f'T2.minimax_radius_m{n}',norm(recover),radius)
        boundary.append(dict(m=n,radius=radius,above_tolerance=bool(radius>=tau)))
    check('T2.exact_boundary',int(boundary[0]['above_tolerance'])+int(not boundary[1]['above_tolerance']),0)
    # Deterministic, feasible midpoint indistinguishability construction.
    P=propagator(pp);H=run(pp)
    Xp=np.linalg.solve(P,endpoint);Xm=-Xp
    close('T2.indistinguishable_pair',P@Xp-endpoint+H,P@Xm+endpoint+H)
    close('T2.minimax_pair_separation',norm(Xp-Xm),2*boundary[-1]['radius'])
    # Ridge gain, bias, and noise bound on an SVD, no data-selected lambda.
    lam=1e-4
    U,sv,Vh=np.linalg.svd(P)
    ridge=(Vh.T*(sv/(sv*sv+lam)))@U.T
    check('T2.ridge_gain',opnorm(ridge)-1/(2*math.sqrt(lam)),2e-11)
    signal=RNG.normal(size=(D,V))*.01
    bias=ridge@(P@signal)-signal
    predicted=Vh.T@((-lam/(sv*sv+lam))[:,None]*(Vh@signal))
    close('T2.ridge_bias',bias,predicted)
    # Checkpoint: replay from a fresh anchor; all remaining errors propagate forward.
    start=m-8
    anchor=h64[start].astype(np.float32)
    _,hh=run(cut(p32,start,m),anchor,states=True)
    err_bound=norm(anchor.astype(np.float64)-h64[start])
    for j in range(8):
        ref=step(hh[j].astype(np.float64),p64['k'][start+j],p64['v'][start+j],p64['beta'][start+j],p64['gamma'][start+j])
        residual=norm(hh[j+1].astype(np.float64)-ref)
        err_bound=float(p64['gamma'][start+j])*err_bound+residual
    anchored=norm(hh[-1].astype(np.float64)-h64[-1])
    check('T2.checkpoint_forward_bound',anchored-err_bound,1e-12)
    eta=2e-7;bar=.99;L=8;e0=.001
    geometric=bar**L*e0+eta*(1-bar**L)/(1-bar)
    recursive=e0
    for _ in range(L):recursive=bar*recursive+eta
    close('T2.checkpoint_geometric_sum',recursive,geometric)
    close('T2.checkpoint_no_decay_limit',e0+L*eta,e0+sum(eta for _ in range(L)))
    check('T2.constructed_natural_breakdown',.1-natural[-1]['rollback_absolute_error'],0)
    # Float32 saturation and underflow are an explicit inapplicability branch.
    sat=float(sigmoid(np.array(20.,dtype=np.float32)))
    under=float(np.exp(np.array(-200.,dtype=np.float32)))
    check('T2.effective_singular_branch',abs(sat-1)+abs(under),0)
    DETAILS['T2']=dict(unit_roundoff=float(u),local_rounding=local,natural_rounding=natural,minimax=dict(delta=delta,tau=tau,m_star=ms,boundary=boundary),checkpoint_error=anchored,checkpoint_bound=err_bound,ridge_lambda=lam,saturated_beta=sat,underflow_gamma=under)


def test_t3():
    trials,m=4096,64;beta=.3;rho=.999;gamma=.999
    rows=[]
    for r in [16,128]:
        # Rank-one d_k x d_v state, represented by its key-side factor for efficiency.
        x=RNG.normal(size=D);x/=norm(x)
        f=np.repeat(x[None,:],trials,axis=0);inv=f.copy()
        for _ in range(m):
            kk=RNG.normal(size=(trials,r));kk/=np.linalg.norm(kk,axis=1,keepdims=True);kk*=rho
            dot=np.sum(kk*f[:,:r],axis=1)
            f[:,:r]-=beta*kk*dot[:,None];f*=gamma
            dot=np.sum(kk*inv[:,:r],axis=1)
            inv[:,:r]+=beta/(1-beta*rho*rho)*kk*dot[:,None];inv/=gamma
        energies=np.sum(f*f,axis=1);inverse_energies=np.sum(inv*inv,axis=1)
        pu=float(np.sum(x[:r]**2));perp=1-pu;b=beta*rho*rho;c=b*(2-b)
        pred=gamma**(2*m)*(perp+(1-c/r)**m*pu)
        pred_inv=gamma**(-2*m)*(perp+(1+((1-b)**-2-1)/r)**m*pu)
        for name,arr,target in [('ghost',energies,pred),('inverse',inverse_energies,pred_inv)]:
            se=float(np.std(arr,ddof=1)/math.sqrt(trials))
            check(f'T3.{name}_expectation_r{r}',abs(float(np.mean(arr))-target),6*se+2e-12)
        half=math.log(2)/(-2*math.log(gamma)-math.log1p(-c/D))
        close(f'T3.half_life_r{r}',(gamma**2*(1-c/D))**half,.5)
        if r<D:
            close('T3.perp_trajectory_exact',f[:,r:],gamma**m*np.repeat(x[None,r:],trials,axis=0))
        rows.append(dict(r=r,trials=trials,m=m,ghost_mean=float(np.mean(energies)),ghost_prediction=pred,ghost_se=float(np.std(energies,ddof=1)/math.sqrt(trials)),inverse_mean=float(np.mean(inverse_energies)),inverse_prediction=pred_inv,inverse_se=float(np.std(inverse_energies,ddof=1)/math.sqrt(trials)),half_life_full_space=half))
    DETAILS['T3']=rows


def compose(left,right):
    # chronological: right after left
    Pa,Ha=left;Pb,Hb=right
    return Pb@Pa,Pb@Ha+Hb


def summary(p):
    return propagator(p),run(p)


class AffineTree:
    def __init__(self,p,C=16):
        self.p=p;self.C=C;self.T=len(p['k']);self.N=(self.T+C-1)//C
        self.size=1
        while self.size<self.N:self.size*=2
        self.nodes=[None]*(2*self.size)
        ident=(np.eye(D,dtype=p['k'].dtype),np.zeros((D,V),dtype=p['k'].dtype))
        for i in range(self.size):
            self.nodes[self.size+i]=summary(cut(p,i*C,min((i+1)*C,self.T))) if i<self.N else ident
        for i in range(self.size-1,0,-1):self.nodes[i]=compose(self.nodes[2*i],self.nodes[2*i+1])

    def node_ids(self,l,r):
        left=[];right=[];l+=self.size;r+=self.size
        while l<r:
            if l&1:left.append(l);l+=1
            if r&1:r-=1;right.append(r)
            l//=2;r//=2
        return left+right[::-1]

    def apply(self,lo,hi,S):
        # Boundary token scans + chronological full-block node applications.
        S=S.copy();count=0;nodes=0
        while lo<hi and lo%self.C:
            S=step(S,self.p['k'][lo],self.p['v'][lo],self.p['beta'][lo],self.p['gamma'][lo]);lo+=1;count+=1
        stop=hi-hi%self.C
        if stop>=lo:
            for node in self.node_ids(lo//self.C,stop//self.C):
                P,H=self.nodes[node];S=P@S+H;nodes+=1
            lo=stop
        while lo<hi:
            S=step(S,self.p['k'][lo],self.p['v'][lo],self.p['beta'][lo],self.p['gamma'][lo]);lo+=1;count+=1
        return S,dict(token_steps=count,dense_nodes=nodes)

    def nbytes(self):
        return sum(P.nbytes+H.nbytes for P,H in self.nodes[1:] if P is not None)

    def propagate_factor(self,lo,hi,L):
        L=L.copy()
        while lo<hi and lo%self.C:
            L=linear(L,self.p['k'][lo],self.p['beta'][lo],self.p['gamma'][lo]);lo+=1
        stop=hi-hi%self.C
        if stop>=lo:
            for node in self.node_ids(lo//self.C,stop//self.C):L=self.nodes[node][0]@L
            lo=stop
        while lo<hi:
            L=linear(L,self.p['k'][lo],self.p['beta'][lo],self.p['gamma'][lo]);lo+=1
        return L


def conv4(raw,weight):
    out=np.zeros_like(raw)
    # weight[lag], lag=0 is current token. Equivalent to causal cross-correlation with reversed weight.
    for lag in range(4):
        out[lag:]+=raw[:len(raw)-lag]*weight[lag]
    return silu(out)


class GDN:
    def __init__(self):
        self.Wq=RNG.normal(size=(D,D))/math.sqrt(D)
        self.Wk=RNG.normal(size=(D,D))/math.sqrt(D)
        self.Wv=RNG.normal(size=(D,V))/math.sqrt(D)
        self.Wz=RNG.normal(size=(D,V))*.2/math.sqrt(D)
        self.Wb=RNG.normal(size=D)*.15/math.sqrt(D)
        self.Wa=RNG.normal(size=D)*.15/math.sqrt(D)
        self.conv=RNG.normal(size=(4,3*D))*.16
        self.conv[0]+=.65
        self.Wout=RNG.normal(size=(V,D))*.15/math.sqrt(D)
        self.W1=RNG.normal(size=(D,2*D))*.15/math.sqrt(D)
        self.Wgate=RNG.normal(size=(D,2*D))*.15/math.sqrt(D)
        self.W2=RNG.normal(size=(2*D,D))*.15/math.sqrt(2*D)

    def operands(self,x,dtype=np.float64):
        x=np.asarray(x,dtype=dtype);xn=rms(x)
        raw=np.concatenate([xn@self.Wq.astype(dtype),xn@self.Wk.astype(dtype),xn@self.Wv.astype(dtype)],axis=-1)
        mixed=conv4(raw,self.conv.astype(dtype))
        q,k,v=np.split(mixed,3,axis=-1)
        q=l2(q,1/math.sqrt(D));k=l2(k)
        b=sigmoid(xn@self.Wb.astype(dtype))
        # A_log=log(.01), dt_bias=0, same gate functional form as Qwen.
        g=-np.array(.01,dtype=dtype)*np.logaddexp(np.array(0,dtype=dtype),xn@self.Wa.astype(dtype))
        return seq(k,v,b,np.exp(g)),q,xn@self.Wz.astype(dtype),raw

    def project(self,o,z,dtype=np.float64):
        return (rms(o)*silu(z))@self.Wout.astype(dtype)

    def post(self,x,dtype=np.float64):
        a=rms(x)
        return x+(silu(a@self.W1.astype(dtype))*(a@self.Wgate.astype(dtype)))@self.W2.astype(dtype)

    def forward(self,x,dtype=np.float64):
        x=np.asarray(x,dtype=dtype);p,q,z,raw=self.operands(x,dtype)
        S=np.zeros((D,V),dtype=dtype);hist=[S.copy()];o=[]
        for k,v,b,g,qt in zip(p['k'],p['v'],p['beta'],p['gamma'],q):
            S=step(S,k,v,b,g);hist.append(S.copy());o.append(S.T@qt)
        o=np.asarray(o)
        y=self.post(x+self.project(o,z,dtype),dtype)
        return y,dict(p=p,q=q,z=z,raw=raw,states=hist,o=o)

    def certified_G(self,T):
        # Analytic domain-wide constants, not sample Jacobians.
        Ln=1/math.sqrt(EPS);root=math.sqrt(D)
        Cq=sum(float(np.max(np.abs(w[:D]))) for w in self.conv)
        Ck=sum(float(np.max(np.abs(w[D:2*D]))) for w in self.conv)
        Cv=sum(float(np.max(np.abs(w[2*D:]))) for w in self.conv)
        Lk=2*Ck*opnorm(self.Wk)*Ln/math.sqrt(EPS)
        Lq=2*Cq*opnorm(self.Wq)*Ln/math.sqrt(EPS*D)
        Lv=2*Cv*opnorm(self.Wv)*Ln
        Lb=.25*norm(self.Wb)*Ln;Lg=.01*norm(self.Wa)*Ln
        Vbound=root*opnorm(self.Wv)*Cv
        Rbound=T*Vbound
        Mz=root*opnorm(self.Wz)
        Lo=opnorm(self.Wout)*Mz/math.sqrt(EPS)
        Lz=2*opnorm(self.Wout)*math.sqrt(V)
        Mpost=1+3*root*opnorm(self.W2)*opnorm(self.W1)*opnorm(self.Wgate)*Ln
        L0=Mpost*(1+Lz*opnorm(self.Wz)*Ln);Lo*=Mpost
        # gamma <= 1 globally; this avoids assuming a trained-head horizon.
        K=Rbound*(Lg+Lb+2*Lk)+Lb*Vbound+Vbound*Lk+Lv
        H=np.zeros((T,T))
        for lag in range(4):
            for t in range(lag,T):H[t,t-lag]=1.
        F=np.tril(np.ones((T,T)))
        return L0*np.eye(T)+Lo*Rbound*Lq*H+Lo/math.sqrt(D)*K*(F@H)


def wy(p):
    W=np.zeros((D,0));U=W.copy()
    for k,b in zip(p['k'],p['beta']):
        W=np.concatenate([W-b*k[:,None]*(k@W)[None,:],(b*k)[:,None]],axis=1)
        U=np.concatenate([U,k[:,None]],axis=1)
    return float(np.prod(p['gamma']))*(np.eye(D)-W@U.T)


def test_t4():
    # Exact terminal-only interior editing. The effective edit is oblique even
    # though every individual GDN contraction is symmetric.
    terminal_rows=[]
    k=(np.eye(D)[0]+np.eye(D)[1])/math.sqrt(2)
    b=.4;g=.99;alpha=b/(1-b);B=b*np.outer(k,np.eye(V)[0])
    for n in [0,4,12,24]:
        eps_s=2.**(-n);decay=.999**n
        diag=np.full(D,decay);diag[0]*=eps_s
        F=np.diag(diag);Fi=np.diag(1/diag)
        w=F@k;z=Fi.T@k
        M=(np.eye(D)+alpha*np.outer(w,z))/g
        J=norm(w)*norm(z);tt=2+2*alpha+alpha**2*J**2
        theta=math.sqrt((tt+math.sqrt(max(0.,tt*tt-4/(1-b)**2)))/2)/g
        uu,sv,vh=np.linalg.svd(M)
        close(f'T4.terminal_edit_singular_n{n}',sv[0],theta)
        close(f'T4.oblique_idempotence_n{n}',np.outer(w,z)@np.outer(w,z),np.outer(w,z))
        X=np.eye(D)*.01
        dirty=F@(g*(np.eye(D)-b*np.outer(k,k))@X+B)
        clean=F@X
        close(f'T4.terminal_edit_identity_n{n}',M@(dirty-F@B),clean,2e-9)
        noise=2.**-24*np.outer(vh[0],np.eye(V)[0])
        # Observe noise separately: no subtraction of huge nearly equal edited states.
        close(f'T4.terminal_edit_minimax_gain_n{n}',norm(M@noise),theta*2.**-24,2e-11)
        ghost=norm(F@B)
        close(f'T4.ghost_and_edit_risk_n{n}',ghost,b*decay*math.sqrt((1+eps_s**2)/2))
        terminal_rows.append(dict(n=n,suffix_anisotropy=1/eps_s,deleted_step_inverse_gain=1/(g*(1-b)),terminal_edit_gain=theta,ghost_norm=ghost,fp32_scale_noise_risk=theta*2.**-24))
    layer=GDN();T=48;x=RNG.normal(size=(T,D))*.2;s,e=12,23 # zero-based inclusive
    keep=np.r_[np.arange(s),np.arange(e+1,T)]
    _,orig=layer.forward(x);_,clean=layer.forward(x[keep])
    p=orig['p'];pc=clean['p'];h=min(T-1,e+3)
    prefix=orig['states'][s]
    halo_clean_indices=np.arange(s,s+(h-e))
    clean_h=run(cut(pc,s,s+h-e),prefix)
    close('T4.halo_state',clean_h,clean['states'][s+h-e])
    close('T4.unchanged_operands_after_halo',p['k'][h+1:],pc['k'][s+h-e:])
    for field in ['v','beta','gamma']:
        close('T4.unchanged_'+field+'_after_halo',p[field][h+1:],pc[field][s+h-e:])
    close('T4.unchanged_query_after_halo',orig['q'][h+1:],clean['q'][s+h-e:])
    for j in range(1,4):
        diff=norm(p['k'][e+j]-pc['k'][s+j-1])
        check(f'T4.halo_token_{j}_needed',1e-5-diff,0)
    ghost=orig['states'][h+1]-clean_h
    Phi=propagator(cut(p,h+1,T))
    close('T4.exact_ghost_formula',orig['states'][-1]-clean['states'][-1],Phi@ghost)
    wrong=run(cut(p,e+1,T),prefix)
    wrong_error=norm(wrong-clean['states'][-1])
    check('T4.missing_halo_negative_control',1e-5-wrong_error,0)
    # Direct clean prefix + halo + suffix, and correction route.
    tree=AffineTree(p,C=8)
    left=np.eye(D)[:,:4];right=np.eye(V)[:,:4]
    close('T4.low_rank_factor_query',tree.propagate_factor(3,43,left)@right.T,propagator(cut(p,3,43))@(left@right.T))
    pfx,_=tree.apply(0,s,np.zeros((D,V)))
    ch=run(cut(pc,s,s+h-e),pfx)
    corrected,counts=tree.apply(h+1,T,ch)
    close('T4.summary_clean_terminal',corrected,clean['states'][-1])
    close('T4.subtract_ghost',orig['states'][-1]-Phi@ghost,clean['states'][-1])
    # A generic non-diagonal suffix, nonzero prefix and affine writes: this
    # core-only single-token intervention deliberately excludes the conv halo.
    j=17;F=propagator(cut(p,j+1,T));Hs=run(cut(p,j+1,T))
    kk=p['k'][j];bb=p['beta'][j];gg=p['gamma'][j];a2=float(kk@kk)
    B=bb*np.outer(kk,p['v'][j]);ww=F@kk;zz=np.linalg.solve(F.T,kk)
    alpha=bb/(1-bb*a2);M=(np.eye(D)+alpha*np.outer(ww,zz))/gg
    cleaned_core=run(cut(p,j+1,T),orig['states'][j])
    close('T4.generic_terminal_edit',M@(orig['states'][-1]-Hs-F@B)+Hs,cleaned_core)
    tt=2+2*alpha*a2+alpha**2*norm(ww)**2*norm(zz)**2
    th=math.sqrt((tt+math.sqrt(max(0.,tt*tt-4/(1-bb*a2)**2)))/2)/gg
    close('T4.generic_terminal_theta',opnorm(M),th)
    # Empty suffix / clipped halo and very short deleted span.
    for ss,ee in [(0,0),(T-4,T-2),(T-3,T-2)]:
        retained=np.r_[np.arange(ss),np.arange(ee+1,T)]
        _,cc=layer.forward(x[retained]);hh=min(T-1,ee+3)
        cpref=orig['states'][ss]
        cle_h=run(cut(cc['p'],ss,ss+hh-ee),cpref)
        ph=propagator(cut(p,hh+1,T))
        close(f'T4.clipped_halo_{ss}_{ee}',orig['states'][-1]-cc['states'][-1],ph@(orig['states'][hh+1]-cle_h))
    close('T4.WY',wy(cut(p,3,20)),propagator(cut(p,3,20)))
    a,b,c=summary(cut(p,0,8)),summary(cut(p,8,16)),summary(cut(p,16,24))
    lhs=compose(compose(a,b),c);rhs=compose(a,compose(b,c))
    close('T4.affine_associativity_P',lhs[0],rhs[0]);close('T4.affine_associativity_H',lhs[1],rhs[1])
    # Arbitrary, empty and short ranges exercise all boundary paths.
    for lo,hi in [(0,0),(3,4),(3,19),(8,40),(45,48),(0,48)]:
        out,ct=tree.apply(lo,hi,prefix)
        close(f'T4.range_{lo}_{hi}',out,run(cut(p,lo,hi),prefix))
        check(f'T4.range_boundary_steps_{lo}_{hi}',ct['token_steps']-2*tree.C,0)
        check(f'T4.range_nodes_{lo}_{hi}',ct['dense_nodes']-2*(math.ceil(math.log2(tree.N))+1),0)
    rankexample=.98*(np.eye(D)-.4*np.outer(np.eye(D)[0],np.eye(D)[0]))-np.eye(D)
    rank=int(np.linalg.matrix_rank(rankexample))
    check('T4.rank_not_span_length',D-rank,0)
    # Float32 vs oracle with same effective operands, not a bitwise WY assertion.
    pp32=cast(p,np.float32);pc32=cast(pc,np.float32)
    tr32=AffineTree(pp32,C=8)
    pre32,_=tr32.apply(0,s,np.zeros((D,V),np.float32))
    hc32=run(cut(pc32,s,s+h-e),pre32)
    terminal32,_=tr32.apply(h+1,T,hc32)
    oracle=run(cast(pc32,np.float64))
    fp32_err=norm(terminal32.astype(np.float64)-oracle)/max(1.,norm(oracle))
    check('T4.float32_summary_oracle',fp32_err,2e-5)
    # Forward residual certificate: a separate finite-precision replay of the clean sequence.
    pp64=cast(pc32,np.float64);_,hs=run(pc32,states=True)
    cert=0.
    for t in range(len(pp64['k'])):
        rerr=norm(hs[t+1].astype(np.float64)-step(hs[t].astype(np.float64),pp64['k'][t],pp64['v'][t],pp64['beta'][t],pp64['gamma'][t]))
        cert=float(pp64['gamma'][t])*cert+rerr
    check('T4.forward_error_certificate',norm(hs[-1].astype(np.float64)-oracle)-cert,1e-12)
    # Query-only benchmark; build, memory, and amortization explicitly paid and reported.
    n=768;kb=l2(RNG.normal(size=(n,D)));pb=seq(kb,RNG.normal(size=(n,V))*.02,np.full(n,.3),np.full(n,.999))
    tic=time.perf_counter();bt=AffineTree(pb,C=16);build=time.perf_counter()-tic
    lo,hi=128,384
    def query():
        a,ca=bt.apply(0,lo,np.zeros((D,V)))
        z,cz=bt.apply(hi,n,a)
        return z,ca,cz
    oracle_b=run(cut(pb,hi,n),run(cut(pb,0,lo)))
    z,ca,cz=query();close('T4.benchmark_accuracy',z,oracle_b)
    def timed(fn):
        vals=[]
        for _ in range(5):
            tt=time.perf_counter();fn();vals.append(time.perf_counter()-tt)
        return float(np.median(vals))
    # Warm both paths before recording medians.
    query();run(cut(pb,hi,n),run(cut(pb,0,lo)))
    tq=timed(query);ts=timed(lambda:run(cut(pb,hi,n),run(cut(pb,0,lo))))
    breakeven=math.ceil(build/(ts-tq)) if ts>tq else None
    archive_bytes=sum(z.nbytes for z in pb.values())
    DETAILS['T4']=dict(T=T,terminal_only_edit=terminal_rows,deleted_zero_based=[s,e],halo_original_zero_based=[e+1,h],missing_halo_error=wrong_error,float32_summary_relative_error=fp32_err,rank_one_span_residual_rank=rank,query_counts=counts,benchmark=dict(T=n,C=16,retained_tokens=n-(hi-lo),build_seconds=build,query_seconds=tq,stream_seconds=ts,observed_query_speedup=ts/tq,amortization_queries=breakeven,tree_bytes=bt.nbytes(),operand_archive_bytes=archive_bytes,query_dense_nodes=ca['dense_nodes']+cz['dense_nodes'],query_boundary_steps=ca['token_steps']+cz['token_steps'],timing_repeats=5,claim='observed query-only time; not universal or end-to-end speedup'))
    return layer,x,orig,clean,keep


def rope(x,positions):
    # Partial RoPE: 64/256 coordinates; orthogonal, so preserves norm/Lipschitz constants.
    nrot=64;half=nrot//2
    freq=1./(1e7**(np.arange(half)/half))
    angle=np.asarray(positions)[:,None,None]*freq[None,None,:]
    c=np.cos(angle);s=np.sin(angle)
    out=x.copy();a=x[...,:half];b=x[...,half:nrot]
    out[...,:half]=a*c-b*s;out[...,half:nrot]=a*s+b*c
    return out


class FullAttention:
    def __init__(self):
        self.F=256;self.H=2 # 2 query heads / 1 KV head, downscaled GQA counts.
        self.Wq=RNG.normal(size=(D,self.H*self.F))*.2/math.sqrt(D)
        self.Wgate=RNG.normal(size=(D,self.H*self.F))*.2/math.sqrt(D)
        self.Wk=RNG.normal(size=(D,self.F))*.2/math.sqrt(D)
        self.Wv=RNG.normal(size=(D,self.F))*.2/math.sqrt(D)
        self.Wout=RNG.normal(size=(self.H*self.F,D))*.15/math.sqrt(self.H*self.F)
        self.W1=RNG.normal(size=(D,2*D))*.15/math.sqrt(D)
        self.Wgate_mlp=RNG.normal(size=(D,2*D))*.15/math.sqrt(D)
        self.W2=RNG.normal(size=(2*D,D))*.15/math.sqrt(2*D)

    def projected(self,x,positions=None):
        xx=rms(x);T=len(x)
        positions=np.arange(T) if positions is None else positions
        q=rope(rms((xx@self.Wq).reshape(T,self.H,self.F)),positions)
        k=rope(rms((xx@self.Wk).reshape(T,1,self.F)),positions)
        v=(xx@self.Wv).reshape(T,1,self.F)
        gate=sigmoid((xx@self.Wgate).reshape(T,self.H,self.F))
        return q,k,v,gate

    def forward(self,x,evict=None,positions=None):
        q,k,v,gate=self.projected(x,positions);T=len(x);outputs=[]
        for t in range(T):
            allowed=np.arange(t+1)
            if evict is not None:allowed=allowed[~np.isin(allowed,evict)]
            if len(allowed)==0:outputs.append(np.zeros(D));continue
            scores=np.einsum('hd,nhd->hn',q[t],np.repeat(k[allowed],self.H,axis=1))/math.sqrt(self.F)
            probs=np.exp(scores-scores.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
            y=(probs@v[allowed,0])*gate[t]
            outputs.append(y.reshape(-1)@self.Wout)
        y=x+np.asarray(outputs)
        a=rms(y)
        return y+(silu(a@self.W1)*(a@self.Wgate_mlp))@self.W2

    def certified_lipschitz(self):
        Ln=1/math.sqrt(EPS);root=math.sqrt(D);F=self.F
        # Bounds after input RMS and per-head RMS; RoPE has operator norm 1.
        Q=K=math.sqrt(F);VF=root*opnorm(self.Wv)
        LK=opnorm(self.Wk)*Ln*Ln;LV=opnorm(self.Wv)*Ln
        vals=[]
        for h in range(self.H):
            LQ=opnorm(self.Wq[:,h*F:(h+1)*F])*Ln*Ln
            Lg=opnorm(self.Wgate[:,h*F:(h+1)*F])*Ln
            vals.append(LV+2*VF/math.sqrt(F)*(Q*LK+K*LQ)+VF*Lg/4)
        post=1+3*root*opnorm(self.W2)*opnorm(self.W1)*opnorm(self.Wgate_mlp)*Ln
        return post*(1+opnorm(self.Wout)*norm(np.array(vals)))


def local_defect_check(label,model,a,b,x,xc):
    # Theorem 5(19), using actual operand differences and a valid whole-trajectory R,V.
    pa,pb=a['p'],b['p'];R=max(norm(S) for S in b['states']);VB=max(np.max(np.linalg.norm(pa['v'],axis=1)),np.max(np.linalg.norm(pb['v'],axis=1)))
    certificate=norm(a['states'][0]-b['states'][0]);max_violation=-math.inf;read_violation=-math.inf
    local_ratios=[]
    for t in range(len(x)):
        dk=norm(pa['k'][t]-pb['k'][t]);db=abs(pa['beta'][t]-pb['beta'][t]);dg=abs(pa['gamma'][t]-pb['gamma'][t]);dv=norm(pa['v'][t]-pb['v'][t])
        at=dg+pb['gamma'][t]*(db+2*pb['beta'][t]*dk)
        bt=db*VB+pb['beta'][t]*(VB*dk+dv)
        certificate=pa['gamma'][t]*certificate+R*at+bt
        actual=norm(a['states'][t+1]-b['states'][t+1]);max_violation=max(max_violation,actual-certificate)
        read=norm(a['o'][t]-b['o'][t]);rb=actual/math.sqrt(D)+R*norm(a['q'][t]-b['q'][t]);read_violation=max(read_violation,read-rb)
        local_ratios.append(actual/max(certificate,1e-30))
    check(label+'.state_defect_bound',max_violation,1e-10)
    check(label+'.raw_read_bound',read_violation,1e-10)
    # Explicit gated RMSNorm Lipschitz constants (22) on the entire segment between z,z'.
    mz=max(float(np.max(np.abs(a['z']))),float(np.max(np.abs(b['z']))))
    Lo=opnorm(model.Wout)*mz/math.sqrt(EPS);Lz=2*opnorm(model.Wout)*math.sqrt(V)
    yo=model.project(a['o'],a['z']);yc=model.project(b['o'],b['z'])
    bound=Lo*np.linalg.norm(a['o']-b['o'],axis=1)+Lz*np.linalg.norm(a['z']-b['z'],axis=1)
    check(label+'.gated_read_bound',float(np.max(np.linalg.norm(yo-yc,axis=1)-bound)),1e-10)
    return dict(state_error=norm(a['states'][-1]-b['states'][-1]),state_certificate=certificate,max_fraction_of_certificate=float(max(local_ratios)))


def test_t5(layer,x,orig,clean,keep):
    layers=[layer,GDN(),GDN()];fa=FullAttention();T=len(x)
    # Same-length edit for the domain-wide causal bound (fixed positional/mask geometry).
    xc=x.copy();xc[12:24]=RNG.normal(size=(12,D))*.2
    dirty=x.copy();edited=xc.copy();d0=np.linalg.norm(dirty-edited,axis=1);bound=d0.copy();Glist=[];records=[]
    full32=x.astype(np.float32)
    for i,model in enumerate(layers):
        din=dirty.copy();cin=edited.copy()
        dirty,a=model.forward(dirty);edited,b=model.forward(edited)
        records.append(local_defect_check(f'T5.layer{i+1}',model,a,b,din,cin))
        G=model.certified_G(T);Glist.append(G);bound=G@bound
        actual=np.linalg.norm(dirty-edited,axis=1)
        check(f'T5.layer{i+1}.analytic_cascade',float(np.max(actual-bound)),1e-8)
        full32,_=model.forward(full32,np.float32)
        if i==0:
            # Actual additive perturbations, not only symbolic composition of b terms.
            base=np.zeros((D,V));injected=base.copy();cert=0.
            kick=np.outer(np.eye(D)[0],np.eye(V)[0])*1e-4
            for t in range(T):
                args=(a['p']['k'][t],a['p']['v'][t],a['p']['beta'][t],a['p']['gamma'][t])
                base=step(base,*args);injected=step(injected,*args)+kick
                cert=float(a['p']['gamma'][t])*cert+norm(kick)
            check('T5.actual_state_injection_bound',norm(injected-base)-cert,1e-11)
    close('T5.three_layer_composition',bound,Glist[2]@Glist[1]@Glist[0]@d0,1e-10)
    # Nonzero injected-state term verifies the inhomogeneous formula in (21).
    bb=[RNG.uniform(0,.01,T) for _ in range(3)]
    recursive=Glist[2]@(Glist[1]@(Glist[0]@d0+bb[0])+bb[1])+bb[2]
    expanded=Glist[2]@Glist[1]@Glist[0]@d0+Glist[2]@Glist[1]@bb[0]+Glist[2]@bb[1]+bb[2]
    close('T5.inhomogeneous_cascade',recursive,expanded,1e-10)
    output=fa.forward(dirty);output_c=fa.forward(edited)
    LF=fa.certified_lipschitz()
    att_actual=float(np.max(np.linalg.norm(output-output_c,axis=1)))
    check('T5.full_attention_bound',att_actual-LF*float(np.max(np.linalg.norm(dirty-edited,axis=1))),1e-8)
    check('T5.whole_block_bound',att_actual-LF*float(np.max(bound)),1e-8)
    # FP32 FA uses float64 fixed toy weights below; all GDN states/operators were FP32.
    # Compare separately to avoid pretending RoPE/softmax used a particular GPU dtype.
    fa32=fa.forward(full32.astype(np.float64))
    fp32_error=norm(fa32-output)/max(1.,norm(output))
    check('T5.three_GDN_fp32_oracle',fp32_error,2e-5)
    # Real sequence deletion exposes cross-layer operand drift.
    original=x;compact=x[keep];drift=[]
    for i,model in enumerate(layers):
        original,aa=model.forward(original);compact,cc=model.forward(compact)
        # Exact fixed-input single-layer deletion: take old layer inputs on retained IDs,
        # but at upper layers these differ from full-network counterfactual inputs.
        if i==0:fixed_input=x[keep]
        else:fixed_input=previous_original[keep]
        fixed_out,ff=model.forward(fixed_input)
        drift.append(dict(layer=i+1,fixed_layer_vs_network_state=norm(ff['states'][-1]-cc['states'][-1]),fixed_layer_vs_network_output=norm(fixed_out-compact)))
        previous_original=original.copy()
    close('T5.first_layer_fixed_equals_replay',drift[0]['fixed_layer_vs_network_state'],0)
    check('T5.upper_layer_drift_negative_control',1e-5-drift[1]['fixed_layer_vs_network_state'],0)
    # KV-only eviction vs compact full block replay, with original positions preserved
    # to avoid attributing the entire effect to RoPE reindexing.
    deleted=np.arange(12,24)
    kvonly=fa.forward(original,evict=deleted)[keep]
    realclean=fa.forward(compact,positions=keep)
    desync=norm(kvonly-realclean)
    check('T5.hybrid_KV_only_desynchronization',1e-5-desync,0)
    # Negative control: a nonzero ghost can have exactly zero readout.
    ghost=np.outer(np.eye(D)[0],np.eye(V)[0]);q=np.eye(D)[1]/math.sqrt(D)
    close('T5.nonzero_ghost_zero_read',ghost.T@q,np.zeros(V))
    # A controlled signed probe certificate, not a claim about learned logits.
    w=np.eye(D)[0];pred=np.zeros(D);pred[0]=-.4;B=.05;true=pred+np.eye(D)[1]*B;cf=.2
    cg=float(w@pred);ct=float(w@true)
    check('T5.signed_certificate_error',abs(ct-cg)-norm(w)*B,1e-12)
    check('T5.signed_conflict',float(cf*ct),0)
    check('T5.signed_margin',norm(w)*B-abs(cg),0)
    DETAILS['T5']=dict(operator='3 single-head GDN + 2-query/1-KV gated FullAttn, head_dim=256, partial_RoPE=.25, conv=4',sequence_length=T,local_bounds=records,cascade_max_bound=float(max(bound)),actual_block_max_difference=att_actual,attention_Lipschitz_constant=LF,three_GDN_float32_relative_error=fp32_error,upper_layer_deletion_drift=drift,KV_only_vs_full_replay_error=desync,conflict='only constructed probe checked; no pretrained-model conflict claim')


def write_report(elapsed,exception=None):
    failed=[x for x in CHECKS if not x['passed']]
    data=dict(seed=SEED,dimensions=dict(d_k=D,d_v=V,conv_width=4,full_attention_head_dim=256),python=sys.version,numpy=np.__version__,platform=platform.platform(),elapsed_seconds=elapsed,total_checks=len(CHECKS),failed_checks=len(failed),status='PASS' if not failed and exception is None else 'FAIL',checks=CHECKS,details=DETAILS,exception=exception,scope='Synthetic CPU verification only. No pretrained weights, GPU, or performance/novelty certification.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path('SMOKE_RESULTS.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
    lines=['# Round 1 CPU Smoke Report','',f"状态：**{data['status']}**；{len(CHECKS)} 项数值检查，失败 {len(failed)} 项。",'',f"固定 seed `{SEED}`；Python `{platform.python_version()}`；NumPy `{np.__version__}`；耗时 {elapsed:.3f}s。",'', '仅依赖 NumPy。GDN d_k=d_v=128，conv W=4；合成 3:1 block 的 FullAttn head_dim=256、partial RoPE=.25，heads/hidden size 缩小。没有加载 Qwen 权重，没有运行 GPU。','', '| 命题 | 数值证据 |','|---|---|']
    if 'T1'in DETAILS:
        a=DETAILS['T1'];lines.append(f"| T1 | 相同 Λ、E：同线逆增益 {a['equal_logs_gains'][0]:.4g}，正交 {a['equal_logs_gains'][1]:.4g}；投影/能量/奇异向量恒等式通过。 |")
    if 'T2'in DETAILS:
        a=DETAILS['T2'];last=a['natural_rounding'][-1];lines.append(f"| T2 | 已知 δ 的 minimax 阈值 m*={a['minimax']['m_star']}；自然 FP32 m=96 误差 {last['rollback_absolute_error']:.4g}，证书 {last['certificate']:.4g}；8 步 checkpoint replay 误差 {a['checkpoint_error']:.4g}。 |")
    if 'T3'in DETAILS:
        for a in DETAILS['T3']:lines.append(f"| T3，r={a['r']} | ghost 均值 {a['ghost_mean']:.6g} vs 期望 {a['ghost_prediction']:.6g}，SE {a['ghost_se']:.3g}；逆能量均值 {a['inverse_mean']:.6g} vs {a['inverse_prediction']:.6g}，SE {a['inverse_se']:.3g}。 |")
    if 'T4'in DETAILS:
        a=DETAILS['T4'];b=a['benchmark'];lines.append(f"| T4 | 省略 halo 错误 {a['missing_halo_error']:.4g}；FP32 summary 相对误差 {a['float32_summary_relative_error']:.4g}；一个被删 token 的残差 rank={a['rank_one_span_residual_rank']}。 |")
        z=a['terminal_only_edit'][-1]
        lines+=['',f"内部编辑反例 n={z['n']}：被删单步逆增益 {z['deleted_step_inverse_gain']:.4g}，终态编辑增益 Θ={z['terminal_edit_gain']:.4g}，ghost 范数 {z['ghost_norm']:.4g}，2⁻²⁴ 终态噪声对应 minimax 风险 {z['fp32_scale_noise_risk']:.4g}。"]
        lines+=['','查询基准（不含 halo 的已捕获操作数 block 删除，合成单 head）：', '',f"- T={b['T']}，保留 {b['retained_tokens']} token；query {b['query_seconds']*1000:.3f}ms，stream {b['stream_seconds']*1000:.3f}ms，观测 query-only 比值 {b['observed_query_speedup']:.2f}×（5 次中位数）。",f"- 构建 {b['build_seconds']:.3f}s；tree {b['tree_bytes']/2**20:.2f} MiB；操作数档案 {b['operand_archive_bytes']/2**20:.2f} MiB；估算摊销盈亏点 {b['amortization_queries']} 次查询。",f"- 查询执行 {b['query_dense_nodes']} 个 dense affine nodes、{b['query_boundary_steps']} 个 token steps。没有证明端到端 GPU speedup。"]
    if 'T5'in DETAILS:
        a=DETAILS['T5'];lines+=['',f"三层固定输入清理与完整网络重前向的终态差：{[round(z['fixed_layer_vs_network_state'],6) for z in a['upper_layer_deletion_drift']]}。KV-only 与完整 replay 差 {a['KV_only_vs_full_replay_error']:.4g}。",'',f"全局解析级联界最大值 {a['cascade_max_bound']:.4g}，实际 block 差 {a['actual_block_max_difference']:.4g}；界很松，只验证有效性，不证明真实模型会放大。"]
    lines+=['','## 完整检查列表','', '| 检查 | 误差/上界违反量 | 容差 | 结果 |','|---|---:|---:|---|']
    for c in CHECKS:lines.append(f"| {c['name']} | {c['error']:.6g} | {c['limit']:.6g} | {'PASS' if c['passed'] else 'FAIL'} |")
    lines+=['','负值表示不等式有余量；恒等式误差使用 scale-aware 归一化。Monte Carlo 采用预先指定 6 SE 容差，不是定理证明。速度不作通过/失败断言。','', '所有数学结论的假设与证明见 FORMULATION.md；文献覆盖与限制见 NOVELTY.md；真实权重预注册见 COLAB_PROBE_PLAN.md。']
    if exception:lines+=['',f'程序异常：`{exception}`']
    Path('SMOKE_REPORT.md').write_text('\n'.join(lines)+'\n')
    return data


def main():
    start=time.perf_counter();exc=None
    try:
        test_t1();test_t2();test_t3();args=test_t4();test_t5(*args)
    except Exception as e:
        import traceback
        traceback.print_exc();exc=repr(e)
    result=write_report(time.perf_counter()-start,exc)
    print(f"\n{result['status']}: {result['total_checks']} checks; {result['failed_checks']} failed; {result['elapsed_seconds']:.3f}s")
    if result['status']!='PASS':sys.exit(1)


if __name__=='__main__':
    main()
