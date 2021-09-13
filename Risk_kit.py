import pandas as pd
import math
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy .optimize import minimize
#from scipy import integrate


def drawdown(return_series:pd.Series):
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdown=(wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth":wealth_index,
                         "Previous_peak":previous_peaks,
                         "Drawdown":drawdown})

def get_ffme_returns():
    me_m=pd.read_csv("Downloads/data/Portfolios_Formed_on_ME_monthly_EW.csv",header=0,index_col=0,na_values=-99.99)
    rets=me_m[['Lo 20','Hi 20']]
    rets.columns=('SmallCap','LargeCap')
    rets=rets/100
    rets.index=pd.to_datetime(rets.index,format="%Y%m").to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC hedge fund index returns
    """
    hfi=pd.read_csv("Downloads/data/edhechedgefundindices.csv",header=0,index_col=0,parse_dates=True)
    hfi=hfi/100
    hfi.index=hfi.index.to_period('M')
    return hfi

def skewness(r):
    demeaned_r=r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    demeaned_r=r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r,level=0.01):
    statistic,p_value=scipy.stats.jarque_bera(r)
    return p_value>level

def get_ind_returns():
    
    ind=pd.read_csv("Downloads/data/ind30_m_vw_rets.csv",header=0,index_col=0)/100
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period('M')
    ind.columns=ind.columns.str.strip()
    return ind

def get_ind_size():
    
    ind=pd.read_csv("Downloads/data/ind30_m_size.csv",header=0,index_col=0)
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period('M')
    ind.columns=ind.columns.str.strip()
    return ind


def get_ind_nfirms():
    
    ind=pd.read_csv("Downloads/data/ind30_m_nfirms.csv",header=0,index_col=0)
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period('M')
    ind.columns=ind.columns.str.strip()
    return ind

def semideviation(r):
    is_negative=r<0
    return r[is_negative].std(ddof=0)

def var_historic(r,level=5):
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    elif isinstance(r,pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError("r is expected to be Series or Daraframe")

def cvar_historic(r,level=5):
    
    if isinstance(r,pd.Series):
        is_beyond= r<= -var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(cvar_historic,level=level)
    else:
        raise TypeError("r is expected to be a series or a dataframe")
        
def var_gaussian(r,level=5,modified=False):
    z=norm.ppf(level/100)
    if modified:
        s=skewness(r)
        k=kurtosis(r)
        z+=((z**2-1)*s/6+
            (z**3-3*z)*(k-3)/24-
            (2*z**3-5*z)*(s**2)/36)
    return -(r.mean()+z*r.std(ddof=0))
        
def annualize_rets(r,periods_per_year):
    compounded_growth=(1+r).prod()
    n_periods=r.shape[0]
    
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r,periods_per_year):
    return r.std(ddof=1)*(periods_per_year**0.5)

def sharpe_ratio(r,riskfree_rate,periods_per_year):
    #convert per year rate to per period
    rf_per_period=(1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret=r-rf_per_period
    ann_ex_ret=annualize_rets(excess_ret,periods_per_year)
    ann_vol=annualize_vol(r,periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_returns(weights, returns):
    """
    Weights to Returns
    """
    return weights.T@returns

def portfolio_vol(weights,covmat):
    """
    Weights To Vol
    """
    return (weights.T@covmat@weights)**0.5

def plot_ef2(n_points,er,cov,style=".-"):
    
    
    if er.shape[0]!=2 or er.shape[0]!=2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights=[np.array([w,1-w]) for w in np.linspace(0,1,n_points)]   
    rets=[portfolio_returns(w,er) for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    import pandas as pd
    ef=pd.DataFrame({"Returns":rets,"Volatility":vols})
    return ef.plot.line(x="Volatility",y="Returns",style=style)


def minimize_vol(target_return,er,cov):
    """
    target_ret->W
    """
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n
    return_is_target={
        'type':'eq',
        'args':(er,),
        'fun': lambda weights, er: target_return-portfolio_returns(weights,er)
    }
    weights_sum_to_1={
        'type':'eq',
        'fun': lambda weights:np.sum(weights)-1
    }
    weights=minimize(portfolio_vol,init_guess,
                    args=(cov,), method="SLSQP",
                     options={'disp':False},
                     constraints=(return_is_target,weights_sum_to_1),
                     bounds=bounds
                    )
    return weights.x

def optimal_weights(n_points,er,cov):
    """
    ->List of weights to run the optimizer on to minimize the vol
    """
    target_rs=np.linspace(er.min(),er.max(),n_points)
    weights=[minimize_vol(target_return,er,cov) for target_return in target_rs]
    return weights


def msr(riskfree_rate,er,cov):
    """
    Riskfree rate + ER + COV -> W
    """
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n
    weights_sum_to_1={
        'type':'eq',
        'fun': lambda weights:np.sum(weights)-1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio,given weights
        """
        r=portfolio_returns(weights,er)
        vol=portfolio_vol(weights,cov)
        return -(r-riskfree_rate)/vol
        
    weights=minimize(neg_sharpe_ratio,init_guess,
                    args=(riskfree_rate,er,cov,), method="SLSQP",
                     options={'disp':False},
                     constraints=(weights_sum_to_1),
                     bounds=bounds
                    )
    
    return weights.x

def gmv(cov):
    """
    Returns the weights of the Global minimum vol portfolio given covariance matrix
    """
    n=cov.shape[0]
    return msr(0,np.repeat(1,n),cov)

def plot_ef(n_points,er,cov,show_cml=False,style=".-",riskfree_rate=0,show_ew=False,show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    weights=optimal_weights(n_points,er,cov)
    rets=[portfolio_returns(w,er) for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({"Returns":rets,"Volatility":vols})
    ax= ef.plot.line(x="Volatility",y="Returns",style=style)
    if show_gmv:
        w_gmv=gmv(cov)
        r_gmv=portfolio_returns(w_gmv,er)
        vol_gmv=portfolio_vol(w_gmv,cov)
        ax.plot([vol_gmv],[r_gmv], color="midnightblue",marker="o",markersize=10)
    
    
    if show_ew:
        n=er.shape[0]
        w_ew=np.repeat(1/n,n)
        r_ew=portfolio_returns(w_ew,er)
        vol_ew=portfolio_vol(w_ew,cov)
        ax.plot([vol_ew],[r_ew], color="goldenrod",marker="o",markersize=12)
    
    if show_cml:
        ax.set_xlim(left=0)
        w_msr=msr(riskfree_rate,er,cov)
        r_msr=portfolio_returns(w_msr,er)
        vol_msr=portfolio_vol(w_msr,cov)

        # Add CML

        cml_x=[0,vol_msr]
        cml_y=[riskfree_rate,r_msr]
        ax.plot(cml_x,cml_y,color="green",marker="o",linestyle="dashed",markersize=12,linewidth=2)
    return ax

def run_cppi(risky_r,safe_r=None,m=3,start=1000,floor=0.8,riskfree_rate=0.03,drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing:Asset Value History, Risk Budget History,Risky Weight History
    """
    #set up the CPPI parameters
    dates=risky_r.index
    n_steps=len(dates)
    account_value=start
    floor_value=start*floor
    peak=start
    if isinstance(risky_r,pd.Series):
        risky_r=pd.DataFrame(risky_r,columns=["R"])
        
    if safe_r is None:
        safe_r=pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:]=riskfree_rate/12 #fast way to set all values to a number 
        
    account_history=pd.DataFrame().reindex_like(risky_r)
    cushion_history=pd.DataFrame().reindex_like(risky_r)
    risky_w_history=pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak=np.maximum(peak,account_value)
            floor_value=peak*(1-drawdown)
        cushion = (account_value-floor_value)/account_value
        risky_w = m*cushion
        risky_w=np.minimum(risky_w,1)
        risky_w=np.maximum(risky_w,0)
        safe_w = 1-risky_w
        risky_alloc=account_value*risky_w
        safe_alloc=account_value*safe_w
        ## update the account for the time step

        account_value=(risky_alloc*(1+risky_r.iloc[step]))+(safe_alloc*(1+safe_r.iloc[step]))
        # save the values so i can look at the history and plot it etc.
        cushion_history.iloc[step]=cushion
        risky_w_history.iloc[step]=risky_w
        account_history.iloc[step]=account_value
        
    
    risky_wealth=start*(1+risky_r).cumprod()
    backtest_result={
        "Wealth":account_history,
        "Risky Wealth":risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation":risky_w_history,
        "m":m,
        "start":start,
        "floor":floor,
        "risky_r":risky_r,
        "safe_r":safe_r
    }
    return backtest_result

def summary_stats(r,riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r=r.aggregate(annualize_rets,periods_per_year=12)
    ann_vol=r.aggregate(annualize_vol,periods_per_year=12)
    ann_sr=r.aggregate(sharpe_ratio,riskfree_rate=riskfree_rate,periods_per_year=12)
    dd=r.aggregate(lambda r:drawdown(r).Drawdown.min())
    skew=r.aggregate(skewness)
    kurt=r.aggregate(kurtosis)
    cf_var5=r.aggregate(var_gaussian,modified=True)
    hist_cvar5=r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return":ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis":kurt,
        "Cornish-Fisher VaR(5%)":cf_var5,
        "Historic CVaR (5%)":hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown":dd
        
    })

def get_total_market_index_returns():
    ind_returns=get_ind_returns()
    ind_nfirms=get_ind_nfirms()
    ind_size=get_ind_size()
    ind_mktcap=ind_nfirms*ind_size
    total_mktcap=ind_mktcap.sum(axis="columns")
    ind_capweight = ind_mktcap.divide(total_mktcap,axis="rows")
    ind_capweight
    total_market_return=(ind_capweight*ind_returns).sum(axis="columns")
    return total_market_return

def gbm(n_years=10, n_scenarios=1000,mu=0.07, sigma=0.15,steps_per_year=12,s_0=100.0,prices=True):
    """
    Evolution of stock prices using a geometric brownian motion model
    
    """
    
    dt=1/steps_per_year
    n_steps=int(n_years*steps_per_year)
    rets_plus_one=np.random.normal(loc=1+mu*dt,scale=sigma*np.sqrt(dt),size=(n_steps+1,n_scenarios))
    rets_plus_one[0]=1
    ret_val=s_0*pd.DataFrame(rets_plus_one).cumprod() if prices else rets_plus_one-1
    return ret_val

def discount(t,r):
    """
    Compute the price of a pure discount bond that pays a dollar at time t, given interest rate r
    """
    if not isinstance(r,pd.DataFrame):
        return (1+r)**(-t)
    discounts=pd.DataFrame([(1+r)**(-i) for i in t])
    discounts.index=t
    return discounts

def pv(l,r):
    """
    Computes present values of a sequence of liabilities
    """
    dates=l.index
    discounts=discount(dates,r)
    return (discounts.multiply(l,axis='rows')).sum()

def funding_ratio(assets, liabilities,r):
    """
    Computes the funding ratio of some assts given liabilities and interest rate
    """
    return pv(assets,r)/pv(liabilities,r)

def show_funding_ratio(assets,r):
    fr=funding_ratio(assets,liabilities,r)
    print(f'{fr*100:.2f}')
    
def cir(n_years=10,n_scenarios=1,a=0.05,b=0.03,sigma=0.05,steps_per_year=12,r_0=None):
    """
    Implements the CIR model for interest rates
    """
    if r_0 is None:
        r_0=b
    r_0=ann_to_inst(r_0)
    dt = 1/steps_per_year
    
    num_steps=int(n_years*steps_per_year)+1
    shock=np.random.normal(0,scale=np.sqrt(dt),size=(num_steps,n_scenarios))
    rates=np.empty_like(shock)
    rates[0]=r_0
    
    h=math.sqrt(a**2+2*sigma**2)
    prices=np.empty_like(shock)
    
    def price(ttm,r):
        _A=((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B=(2*(math.exp(h*ttm)-1))/(2*h+(h+a)*(math.exp(h*ttm)-1))
        _P=_A*np.exp(-_B*r)
        return _P
    prices[0]=price(n_years,r_0)
    
    for step in range(1,num_steps):
        r_t=rates[step-1]
        d_r_t=a*(b-r_t)*dt+ sigma*np.sqrt(r_t)*shock[step]
        rates[step]=abs(r_t+d_r_t)
        prices[step]=price(n_years-step*dt,rates[step])

    rates=pd.DataFrame(data=inst_to_ann(rates),index=range(num_steps))
    
    prices=pd.DataFrame(data=prices,index=range(num_steps))
    
    return rates,prices

def inst_to_ann(r):
    """
    Converts short rate to an annualized rate
    """
    return np.expm1(r)


def ann_to_inst(r):
    
    return np.log1p(r)

def bond_cash_flows(maturity,principal=100,coupon_rate=0.03,coupons_per_year=12):
    """
    Returns a series of cash flow generated by a bond,
    indexed by a coupon number
    """
    n_coupons=round(maturity*coupons_per_year)
    coupon_amt=principal*coupon_rate/coupons_per_year
    coupon_times=np.arange(1,n_coupons+1)
    cash_flows=pd.Series(data=coupon_amt,index=coupon_times)
    cash_flows.iloc[-1]+=principal
    return cash_flows

def bond_price(maturity,principal=100,coupon_rate=0.03,coupons_per_year=12,discount_rate=0.03):
    
    if isinstance(discount_rate,pd.DataFrame):
        pricing_dates=discount_rate.index
        prices=pd.DataFrame(index=pricing_dates,columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t]=bond_price(maturity-t/coupons_per_year,principal,coupon_rate,coupons_per_year,
                                     discount_rate.loc[t])
        return prices
    else:
        if maturity<=0:return principal+principal*coupon_rate/coupons_per_year
        cash_flows=bond_cash_flows(maturity,principal,coupon_rate,coupons_per_year)
        return pv(cash_flows,discount_rate/coupons_per_year)

def macaulay_duration(flows,discount_rate):
    discounted_flows=discount(flows.index,discount_rate)*flows
    weights =discounted_flows/discounted_flows.sum()
    return np.average(flows.index,weights=weights)

def match_durations(cf_t,cf_s,cf_l,discount_rate):
    d_t=macaulay_duration(cf_t,discount_rate)
    d_s=macaulay_duration(cf_s,discount_rate)
    d_l=macaulay_duration(cf_l,discount_rate)
    return (d_l-d_t)/(d_l-d_s)

def bond_total_return(monthly_prices,principal,coupon_rate,coupons_per_year):
    coupons=pd.DataFrame(data=0,index=monthly_prices.index,columns=monthly_prices.columns)
    t_max=monthly_prices.index.max()
    pay_date=np.linspace(12/coupons_per_year,t_max,int(coupons_per_year*t_max/12),dtype=int)
    coupons.iloc[pay_date]=principal*coupon_rate/coupons_per_year
    total_returns=(monthly_prices+coupons)/monthly_prices.shift()-1
    return total_returns.dropna()


def bt_mix(r1,r2,allocator, **kwargs):
    """
    Runs a bactest of allocating beteween two sets of returns
    r1 and r2 are TxN dataframes or returns where T is the time step index and N is the number of scenarios.
    """
    if not r1.shape==r2.shape:
        raise ValueError("r1 and r2 need to be of the same shape")
    weights=allocator(r1,r2,**kwargs)
    if not weights.shape==r1.shape:
        raise ValueErroe("Allocator returns shape that is not same as r1")
    r_mix=weights*r1+(1-weights)*r2
    return r_mix


def fixedmix_allocator(r1,r2,w1,**kwargs):
    """
    returns an T x N Dataframe of psp weights
    """
    return pd.DataFrame(data=w1,index=r1.index,columns=r1.columns)

def terminal_values(rets):
    """
    Returns the final values at the end of the return period for each scenario
    """
    return (rets+1).prod()

def terminal_stats(rets,floor=0.8,cap=np.inf,name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N dataframe of returns, where T is the time-step(we assume rets is sorted by time)
    Returns a 1 column dataframe of Summary Stats indexed by the stat name
    """
    terminal_wealth=(rets+1).prod()
    breach=terminal_wealth<floor
    reach=terminal_wealth>=cap
    p_breach=breach.mean() if breach.sum() > 0 else np.nan
    p_reach=reach.mean() if reach.sum()>0 else np.nan
    e_short=(floor-terminal_wealth[breach]).mean() if breach.sum()>0 else np.nan
    e_surplus=(cap-terminal_wealth[reach]).mean() if reach.sum()>0 else np.nan
    sum_stats=pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short": e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index",columns=[name])
    return sum_stats

def glidepath_allocator(r1,r2,start_glide=1,end_glide=0):
    """
    Simulates a Target-Date-Fund style gradual move from r1 to r2
    """
    n_points=r1.shape[0]
    n_col=r1.shape[1]
    path=pd.Series(data=np.linspace(start_glide,end_glide,num=n_points))
    paths=pd.concat([path]*n_col,axis=1)
    paths.index=r1.index
    paths.columns=r1.columns
    return paths

def floor_allocator(psp_r,ghp_r,floor,zc_prices,m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorith by investing a multiple of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the psp
    """
    if zc_prices.shape!=psp_r.shape:
        raise ValueError("PSP and ZC prices must have the same shape")
    n_steps,n_scenarios=psp_r.shape
    account_value=np.repeat(1,n_scenarios)
    floor_value=np.repeat(1,n_scenarios)
    w_history=pd.DataFrame(index=psp_r.index,columns=psp_r.columns)
    for step in range(n_steps):
        floor_value=floor*zc_prices.iloc[step]
        cushion=(account_value-floor_value)/account_value
        psp_w=(m*cushion).clip(0,1)#same as applying min and max
        ghp_w=1-psp_w
        psp_alloc=account_value*psp_w
        ghp_alloc=account_value*ghp_w
        account_value=psp_alloc*(1+psp_r.iloc[step]+ghp_alloc*(1+ghp_r.iloc[step]))
        w_history.iloc[step]=psp_w
    return w_history

def drawdown_allocator(psp_r,ghp_r,maxdd,m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorith by investing a multiple of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the psp
    """
    n_steps,n_scenarios=psp_r.shape
    account_value=np.repeat(1,n_scenarios)
    floor_value=np.repeat(1,n_scenarios)
    peak_value=np.repeat(1,n_scenarios)
    w_history=pd.DataFrame(index=psp_r.index,columns=psp_r.columns)
    for step in range(n_steps):
        floor_value=(1-maxdd)*peak_value
        cushion=(account_value-floor_value)/account_value
        psp_w=(m*cushion).clip(0,1)#same as applying min and max
        ghp_w=1-psp_w
        psp_alloc=account_value*psp_w
        ghp_alloc=account_value*ghp_w
        account_value=psp_alloc*(1+psp_r.iloc[step]+ghp_alloc*(1+ghp_r.iloc[step]))
        peak_value=np.maximum(peak_value,account_value)
        w_history.iloc[step]=psp_w
    return w_history