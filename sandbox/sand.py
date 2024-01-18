#!/usr/bin/env python

# In[1]:


import os

os.chdir("/Users/jeffnewman/LocalGit/airtaxi-airport")
import sys

sys.path.insert(0, "/Users/jeffnewman/LocalGit/airtaxi-airport")

import airtaxi_airport as aa
import numpy as np

# In[2]:


os.getcwd()

# In[3]:


biz_data = aa.data.business_data()

# In[4]:


m0 = aa.business_models.basic_model(biz_data)

# In[5]:


m0.unmangle()

# In[6]:


simple = m0.copy()

# In[7]:


# r_simple = simple.jax_maximize_loglike(stderr=True)
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Larch Model Dashboard                                                             ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ optimization complete                                                             │
# │ Log Likelihood Current =     -19354.101562 Best =     -19354.101562               │
# │ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓ │
# │ ┃ Parameter                      ┃ Estimate       ┃ Std. Error     ┃ t-Stat     ┃ │
# │ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩ │
# │ │ AIRTAXI_ASC                    │ -0.037228206   │  0.12279169    │ -0.3032    │ │
# │ │ AIRTAXI_Automation             │ -0.0085035238  │  0.033995874   │ -0.2501    │ │
# │ │ AIRTAXI_FlewBizFirst           │  0.79608297    │  0.042879771   │  18.57     │ │
# │ │ AIRTAXI_FlewPrCoach            │  0.61892388    │  0.045808922   │  13.51     │ │
# │ │ AIRTAXI_OTCat4                 │ -0.019597662   │  0.040650524   │ -0.4821    │ │
# │ │ AIRTAXI_RSFreq123              │  0.86330545    │  0.067486577   │  12.79     │ │
# │ │ AIRTAXI_RSFreq4                │  0.4286364     │  0.078318194   │  5.473     │ │
# │ │ AIRTAXI_RideGuarantee          │  0.19968689    │  0.036438972   │  5.48      │ │
# │ │ AIRTAXI_Time                   │ -0.00026781872 │  0.0026393034  │ -0.1015    │ │
# │ │ ANYMODE_Cost_Total             │ -0.00078141051 │  0.00036265806 │ -2.155     │ │
# │ │ AUTO_I_ParkAirport             │  0.27328832    │  0.015816987   │  17.28     │ │
# │ │ AUTO_Male                      │  0.18623255    │  0.041989055   │  4.435     │ │
# │ │ AUTO_ParkCat1                  │  0.049715561   │  0.039015297   │  1.274     │ │
# │ │ AUTO_Time                      │ -0.00053146968 │  0.0015620907  │ -0.3402    │ │
# │ │ SR_ASC                         │  0.2609019     │  0.1229628     │  2.122     │ │
# │ │ SR_Automation                  │ -0.21353463    │  0.030790085   │ -6.935     │ │
# │ │ SR_I_RSAirport                 │  0.15656689    │  0.016410669   │  9.541     │ │
# │ │ SR_I_RSDistance                │ -0.032926934   │  0.015673213   │ -2.101     │ │
# │ │ SR_RSFreq123                   │  1.3581689     │  0.070902474   │  19.16     │ │
# │ │ SR_RSFreq4                     │  0.96154993    │  0.075679399   │  12.71     │ │
# │ │ SR_Time                        │ -0.0057215029  │  0.0017691301  │ -3.234     │ │
# │ └────────────────────────────────┴────────────────┴────────────────┴────────────┘ │
# └───────────────────────────────────────────────────────────────────────────────────┘


# In[8]:


nt_model = aa.latent_class_non_trader(m0, biz_data)

# ### Latent Casewise

# In[9]:


nt_model.jax_loglike_casewise(nt_model.pvals)

# In[10]:


nt_model.pvals = {
    "AIRTAXI_ASC": -0.17443869277157972,
    "AIRTAXI_Automation": -0.06072019610437766,
    "AIRTAXI_FlewBizFirst": 0.6212327774278571,
    "AIRTAXI_FlewPrCoach": 0.46904019636061667,
    "AIRTAXI_OTCat4": -0.023784771031072485,
    "AIRTAXI_RSFreq123": 0.45695320795364053,
    "AIRTAXI_RSFreq4": 0.08209171966002254,
    "AIRTAXI_RideGuarantee": 0.28916999367465684,
    "AIRTAXI_Time": -0.0011128267274963827,
    "ANYMODE_Cost_Total": -(0.003537405347599533),
    "AUTO_I_ParkAirport": 0.20551731167886292,
    "AUTO_Male": 0.0564899173692452,
    "AUTO_ParkCat1": -0.0076732946513516686,
    "AUTO_Time": -0.00849744774221313,
    "SR_ASC": 0.030519594165332724,
    "SR_Automation": -0.3004062310002925,
    "SR_I_RSAirport": 0.1116174971178979,
    "SR_I_RSDistance": -0.034450460809031476,
    "SR_RSFreq123": 0.9911743309045934,
    "SR_RSFreq4": 0.6695528077561699,
    "SR_Time": -0.006211167141351883,
    "_NT_AIRTAXI": -2.4506250423285794,
    "_NT_AUTO": -2.4973467748004814,
    "_NT_COST_TOTAL": -2.7189461990800803,
    "_NT_SR": -2.4639440029473296,
    "_NT_TIME": -5.598850269655563,
    "_NonTraderConst": 500.0,
    "_NonTraderVar": -500.0,
}

# #### 100

# In[11]:


A1 = np.log(
    np.where(
        nt_model.dataset["ch"],
        nt_model._models[100].jax_probability(nt_model.pvals),
        1.0,
    ).prod((1, 2))
)
A1

# In[12]:


A2 = np.log(
    np.where(
        nt_model.dataset["ch"],
        nt_model._models[201].jax_probability(nt_model.pvals),
        1.0,
    ).prod((1, 2))
)

# In[13]:


np.log(
    nt_model._models["classmodel"].jax_probability(nt_model.pvals)[0, 0, 0] * np.exp(A1)
    + nt_model._models["classmodel"].jax_probability(nt_model.pvals)[0, 0, 1]
    * np.exp(A2)
)

# In[ ]:


# In[14]:


# r_nt_model = nt_model.jax_maximize_loglike(stderr=True)
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Larch Model Dashboard                                                             ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ optimization complete                                                             │
# │ Log Likelihood Current =     -16499.847656 Best =     -16499.847656               │
# │ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓ │
# │ ┃ Parameter                      ┃ Estimate       ┃ Std. Error     ┃ t-Stat     ┃ │
# │ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩ │
# │ │ AIRTAXI_ASC                    │ -0.17586388    │  0.0072023901  │ -24.42     │ │
# │ │ AIRTAXI_Automation             │ -0.060554015   │  0.035232469   │ -1.719     │ │
# │ │ AIRTAXI_FlewBizFirst           │  0.62059213    │  0.024185833   │  25.66     │ │
# │ │ AIRTAXI_FlewPrCoach            │  0.46798547    │  0.022013396   │  21.26     │ │
# │ │ AIRTAXI_OTCat4                 │ -0.023433076   │  0.017865064   │ -1.312     │ │
# │ │ AIRTAXI_RSFreq123              │  0.45713203    │  0.015549161   │  29.4      │ │
# │ │ AIRTAXI_RSFreq4                │  0.082536599   │  0.011844836   │  6.968     │ │
# │ │ AIRTAXI_RideGuarantee          │  0.28881151    │  0.026830224   │  10.76     │ │
# │ │ AIRTAXI_Time                   │ -0.0011099985  │  0.0026908277  │ -0.4125    │ │
# │ │ ANYMODE_Cost_Total             │ -0.003539899   │  0.00042716091 │ -8.287     │ │
# │ │ AUTO_I_ParkAirport             │  0.20529413    │  0.014756314   │  13.91     │ │
# │ │ AUTO_Male                      │  0.056079922   │  0.0075625973  │  7.415     │ │
# │ │ AUTO_ParkCat1                  │ -0.0077062645  │  0.011314987   │ -0.6811    │ │
# │ │ AUTO_Time                      │ -0.0085164144  │  0.0016267631  │ -5.235     │ │
# │ │ SR_ASC                         │  0.029400397   │  0.0030133633  │  9.757     │ │
# │ │ SR_Automation                  │ -0.30016058    │  0.033389144   │ -8.99      │ │
# │ │ SR_I_RSAirport                 │  0.11164246    │  0.01624961    │  6.87      │ │
# │ │ SR_I_RSDistance                │ -0.034369505   │  0.017596396   │ -1.953     │ │
# │ │ SR_RSFreq123                   │  0.99001618    │  0.016886951   │  58.63     │ │
# │ │ SR_RSFreq4                     │  0.66822536    │  0.016458802   │  40.6      │ │
# │ │ SR_Time                        │ -0.0062198811  │  0.0017718086  │ -3.51      │ │
# │ │ _NT_AIRTAXI                    │ -2.4522489     │  1.4413808e-05 │ -1.701e+05 │ │
# │ │ _NT_AUTO                       │ -2.4982533     │  2.7775801e-05 │ -8.994e+04 │ │
# │ │ _NT_COST_TOTAL                 │ -2.7197702     │  0.00013947256 │ -1.95e+04  │ │
# │ │ _NT_SR                         │ -2.4634011     │  0.00013080958 │ -1.883e+04 │ │
# │ │ _NT_TIME                       │ -5.5842112     │  1.4262851e-05 │ -3.915e+05 │ │
# │ │ _NonTraderConst                │  24.981522     │  locked        │  locked    │ │
# │ │ _NonTraderVar                  │ -24.999993     │  locked        │  locked    │ │
# │ └────────────────────────────────┴────────────────┴────────────────┴────────────┘ │
# └───────────────────────────────────────────────────────────────────────────────────┘


# In[15]:


m1 = aa.business_models.mix_cost(
    biz_data, n_draws=300, seed=42, prerolled_draws=True, common_draws=True
)

# In[16]:


m1.pvals = {"ANYMODE_Cost_Total": np.log(0.00078141051), "ANYMODE_Cost_Total_s": 0.1}
m1.jax_loglike(m1.pvals)

# In[17]:


m1.set_cap()
m1.pf

# In[18]:


# r1 = m1.jax_maximize_loglike(stderr=True)
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Larch Model Dashboard                                                             ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ optimization complete                                                             │
# │ Log Likelihood Current =     -19349.222656 Best =     -19349.222656               │
# │ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓ │
# │ ┃ Parameter                      ┃ Estimate       ┃ Std. Error     ┃ t-Stat     ┃ │
# │ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩ │
# │ │ AIRTAXI_ASC                    │ -0.062037895   │  0.12422625    │ -0.4994    │ │
# │ │ AIRTAXI_Automation             │ -0.0037884525  │  0.034389284   │ -0.1102    │ │
# │ │ AIRTAXI_FlewBizFirst           │  0.80607994    │  0.04359401    │  18.49     │ │
# │ │ AIRTAXI_FlewPrCoach            │  0.6283452     │  0.046492144   │  13.52     │ │
# │ │ AIRTAXI_OTCat4                 │ -0.0373672     │  0.041879736   │ -0.8923    │ │
# │ │ AIRTAXI_RSFreq123              │  0.86192026    │  0.067887142   │  12.7      │ │
# │ │ AIRTAXI_RSFreq4                │  0.4259057     │  0.078789845   │  5.406     │ │
# │ │ AIRTAXI_RideGuarantee          │  0.19608612    │  0.036892354   │  5.315     │ │
# │ │ AIRTAXI_Time                   │  0.00070691059 │  0.0026870228  │  0.2631    │ │
# │ │ ANYMODE_Cost_Total             │ -8.3749252     │  0.86197066    │ -9.716     │ │
# │ │ ANYMODE_Cost_Total_s           │  2.2724165     │  0.56797904    │  4.001     │ │
# │ │ AUTO_I_ParkAirport             │  0.27351314    │  0.015923385   │  17.18     │ │
# │ │ AUTO_Male                      │  0.18848541    │  0.042276189   │  4.458     │ │
# │ │ AUTO_ParkCat1                  │  0.056759922   │  0.039427783   │  1.44      │ │
# │ │ AUTO_Time                      │ -0.00098544512 │  0.001582692   │ -0.6226    │ │
# │ │ SR_ASC                         │  0.13727748    │  0.1354803     │  1.013     │ │
# │ │ SR_Automation                  │ -0.22389041    │  0.031765077   │ -7.048     │ │
# │ │ SR_I_RSAirport                 │  0.16075807    │  0.016911099   │  9.506     │ │
# │ │ SR_I_RSDistance                │ -0.034959525   │  0.016058484   │ -2.177     │ │
# │ │ SR_RSFreq123                   │  1.3820589     │  0.074356131   │  18.59     │ │
# │ │ SR_RSFreq4                     │  0.98381727    │  0.078737058   │  12.49     │ │
# │ │ SR_Time                        │ -0.0047079866  │  0.0018483906  │ -2.547     │ │
# │ └────────────────────────────────┴────────────────┴────────────────┴────────────┘ │
# └───────────────────────────────────────────────────────────────────────────────────┘


# In[19]:


m1.copy().pf

# In[20]:


nt_mix_model = aa.latent_class_non_trader(m1.copy(), biz_data)

# In[21]:


nt_mix_model.pf

# In[22]:


nt_mix_model.make_random_draws(engine="jax")

# In[23]:


nt_mix_model.jax_loglike(nt_mix_model.pvals)

# In[ ]:


nt_mix_model.pvals = {"ANYMODE_Cost_Total": -8, "ANYMODE_Cost_Total_s": 0.01}

# In[ ]:


LC_params_1 = {
    "AIRTAXI_ASC": -0.17443869277157972,
    "AIRTAXI_Automation": -0.06072019610437766,
    "AIRTAXI_FlewBizFirst": 0.6212327774278571,
    "AIRTAXI_FlewPrCoach": 0.46904019636061667,
    "AIRTAXI_OTCat4": -0.023784771031072485,
    "AIRTAXI_RSFreq123": 0.45695320795364053,
    "AIRTAXI_RSFreq4": 0.08209171966002254,
    "AIRTAXI_RideGuarantee": 0.28916999367465684,
    "AIRTAXI_Time": -0.0011128267274963827,
    "ANYMODE_Cost_Total": np.log(0.003537405347599533),
    "ANYMODE_Cost_Total_s": 0.0,
    "AUTO_I_ParkAirport": 0.20551731167886292,
    "AUTO_Male": 0.0564899173692452,
    "AUTO_ParkCat1": -0.0076732946513516686,
    "AUTO_Time": -0.00849744774221313,
    "SR_ASC": 0.030519594165332724,
    "SR_Automation": -0.3004062310002925,
    "SR_I_RSAirport": 0.1116174971178979,
    "SR_I_RSDistance": -0.034450460809031476,
    "SR_RSFreq123": 0.9911743309045934,
    "SR_RSFreq4": 0.6695528077561699,
    "SR_Time": -0.006211167141351883,
    "_NT_AIRTAXI": -2.4506250423285794,
    "_NT_AUTO": -2.4973467748004814,
    "_NT_COST_TOTAL": -2.7189461990800803,
    "_NT_SR": -2.4639440029473296,
    "_NT_TIME": -5.598850269655563,
    "_NonTraderConst": 500.0,
    "_NonTraderVar": -500.0,
}

# In[ ]:


nt_mix_model.pmaximum = {"ANYMODE_Cost_Total": -1.0, "ANYMODE_Cost_Total_s": 25.0}

# In[ ]:


nt_mix_model.pvals = LC_params_1

# In[ ]:


nt_mix_model.pf

# In[ ]:


nt_mix_model.jax_loglike(nt_mix_model.pvals)

# ### Mixed Casewise

# In[ ]:


nt_mix_model.jax_loglike_casewise(nt_mix_model.pvals)

# #### 100

# In[ ]:


np.log(
    np.where(
        nt_mix_model.dataset["ch"],
        nt_mix_model._models[100].jax_probability(B0),
        1.0,
    ).prod((1, 2))
)

# In[ ]:


np.log(
    np.where(
        nt_mix_model.dataset["ch"],
        nt_mix_model._models[201].jax_probability(B0),
        1.0,
    ).prod((1, 2))
)

# In[ ]:


nt_mix_model._models[201].jax_probability(B0)

# In[ ]:


B0 = nt_mix_model.apply_random_draws(nt_mix_model.pvals, nt_mix_model._draws)[0, 0, :]
B1 = nt_mix_model.apply_random_draws(nt_mix_model.pvals, nt_mix_model._draws)[1, 0, :]

# In[ ]:


nt_mix_model.dataset

# In[ ]:


stop

# In[ ]:


r1mix = nt_mix_model.jax_maximize_loglike(stderr=True)

# In[ ]:


nt_mix_model.dataset

# In[ ]:


nt_mix_model._models[100].dataset

# In[ ]:


nt_mix_model.jax_probability(nt_mix_model.pvals).shape

# In[ ]:


nt_mix_model.dataset["ch"].shape

# In[ ]:


m1.mixtures[0].roll(
    m1._draws[:, 0],
)

# In[ ]:


m1.pvals = {"Cost_Total": 2.0, "Cost_Total_s": 0.0}

# In[ ]:


m1.jax_loglike(m1.pvals)

# In[ ]:


m1.utility_ca

# In[ ]:


m1.apply_random_draws(m1.pvals, m1._draws)[:, 14]

# In[ ]:


m1.pnames[14]

# In[ ]:
