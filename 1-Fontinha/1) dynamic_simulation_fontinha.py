# %% Funções
import epamodule as en
import numpy as np
from random import randint, uniform,choices, seed
import pandas as pd
import matplotlib.pyplot as plt
import tools.helper as helper
import scienceplots
import os 
import re
import seaborn as sns

def generate_constant_pattern(constant_val,idx):

    my_pattern_index=int(en.ENgetnodevalue(idx,en.EN_PATTERN))
    
    #Get pattern length
    pattern_length=en.ENgetpatternlen(my_pattern_index)
    for period in range(1, pattern_length + 1):
        en.ENsetpatternvalue(my_pattern_index, period, constant_val)
        
def tank_indexes():
    n_links = en.ENgetcount(en.EN_NODECOUNT)
    tank_idx = []
    junction_idx = []
    for i in range(1, n_links + 1):
        type = en.ENgetnodetype(i)
        if type == en.EN_TANK:
            tank_idx.append(i)
        if type == en.EN_JUNCTION:
            junction_idx.append(i)
    #print('junction index', junction_idx)
    return tank_idx, junction_idx

def junction_base_demand():
    junction_demand = dict()
    index= []
    demand_node = []
    for i in range(0,len(junction_idx)):
        if en.ENgetnodevalue(junction_idx[i], en.EN_BASEDEMAND)>0:
            index.append(i)
            demand_node.append(en.ENgetnodevalue(junction_idx[i], en.EN_BASEDEMAND))
            junction_demand["junction_"+str(i)] = []#en.ENgetnodevalue(junction_idx[i], en.EN_BASEDEMAND)
    #print('index node ', index)
    return junction_demand,index,demand_node

def pump_indexes():
    n_links = en.ENgetcount(en.EN_LINKCOUNT)
    pump_idx = []
    for i in range(1, n_links + 1):
        type = en.ENgetlinktype(i)
        if type == en.EN_PUMP:
             pump_idx.append(i)
    return pump_idx

def dic_tanks():
    tanks= {}
    tank_init_level = []
    tanks_final = {}
    for x in range(0,len(tank_idx)):
        tanks["tank_"+str(x)] =[]
        tank_init_level.append(en.ENgetnodevalue(tank_idx[x], en.EN_TANKLEVEL))
        tanks_final["tank_final_"+str(x)] = []
    return tanks,tank_init_level,tanks_final


def dic_pumps():
    pumps = {}
    pump = {}
    for x in range(0,len(pump_idx)):
       pumps["pumps_"+str(x)] =[]
       pump["pumps_kw_"+str(x)] = []
    return pumps, pump

def append_tanks():
    for x in range(0,len(tank_idx)):
        tanks["tank_"+str(x)].append(round(en.ENgetnodevalue(tank_idx[x], en.EN_PRESSURE),4))
    return tanks

def append_final_tanks():
    if t_step!=0:
        for x in range(0,len(tank_idx)):
            tanks_final["tank_final_"+str(x)].append(round(en.ENgetnodevalue(tank_idx[x], en.EN_PRESSURE),4)) 
    
    if t_step==0:
        for x in range(0,len(tank_idx)):
            tanks_final["tank_final_"+str(x)].append(0)
    return tanks_final

def append_pumps():
    for x in range(0,len(pump_idx)):
        pumps["pumps_"+str(x)].append(round(en.ENgetlinkvalue(pump_idx[x],en.EN_STATUS),2))
    
    return pumps    

# def append_flow():
#     #for x in range(0,len(pump_idx)):
#     flow["Flow"].append(round(en.ENgetlinkvalue(4,en.EN_FLOW),2))
    
#     return flow

def pump_kw():
    for x in range(0,len(pump_idx)):
        pump["pumps_kw_"+str(x)].append(round(en.ENgetlinkvalue(pump_idx[x],en.EN_ENERGY),2))
    return pump

def total_price_hour(price_pump_hour):   # price_pump_hour,is a dictionary that contains the hourly prices of each pump.
    l = np.zeros(24) # This initializes a NumPy array l with 24 zeros, representing the 24 hours in a day.
    for i in range(0,len(pump_idx)):
        l += price_pump_hour["price_pump_"+str(i)]
    return l

def total_price(price_per_hour):
    return sum(price_per_hour) #Sum as preços das 24 horas e da o custo total daquele dia

def demand_nodes():
    demand = 0
    for i in range(0,len(junction_idx)):
        demand += en.ENgetnodevalue(junction_idx[i],en.EN_DEMAND)
        #print(demand)
    demand_hour.append(demand)
    return demand_hour


def set_pump_state():  # Definir o estado da bomba aleatoriamente
    for x in range(0, len(pump_idx)):
        for j in range(0, len(tank_idx)):
            if len(tanks["tank_"+str(j)]) == 0:
                 en.ENsetlinkvalue(pump_idx[x], en.EN_STATUS, randint(0, 1))
            elif tanks["tank_"+str(j)][-1] < 2: 
                 en.ENsetlinkvalue(pump_idx[x], en.EN_STATUS, 1)  # liga
            elif tanks["tank_"+str(j)][-1] > 20:  # Quero um grande range de daods para treinar o modelo
                en.ENsetlinkvalue(pump_idx[x], en.EN_STATUS, 0)  # desliga
            else:
                en.ENsetlinkvalue(pump_idx[x], en.EN_STATUS, randint(0, 1))
        
def set_initial_level_tank():
    for m in range(0,len(tank_idx)):
        value = en.ENgetnodevalue(tank_idx[m], en.EN_TANKLEVEL)
        #lim = 0.5 # fontinha 1# 
        min_tank_level = 2# en.ENgetnodevalue(tank_idx[m], en.EN_MINLEVEL)
        max_tank_level = 15 # en.ENgetnodevalue(tank_idx[m], en.EN_MAXLEVEL)
        en.ENsetnodevalue(tank_idx[m], en.EN_TANKLEVEL, uniform(min_tank_level, max_tank_level)) # value*uniform(0.75, 1.25) novo nivel será maior ou menor 25% que o outro
                                                        # richmond (del value) uniform(min_tank_level + lim, max_tank_level - lim

def set_initial_demand_value():
    for n in index: 
        value = en.ENgetnodevalue(junction_idx[n], en.EN_BASEDEMAND)*uniform(0.75,1.2) 
        en.ENsetnodevalue(junction_idx[n], en.EN_BASEDEMAND,value) 
      
        
        
def demand_value():
    for n in index:
        value = en.ENgetnodevalue(junction_idx[n], en.EN_DEMAND)
        junction_demand["junction_"+str(n)].append(value)
    return junction_demand

def restore_demand_value():
    for n in range(0,len(index)):
        en.ENsetnodevalue(junction_idx[index[n]], en.EN_BASEDEMAND, demand_node[n]) 
    """
    function restores the initial demand values for the junction nodes in the water network model. The function 
    sets the demand values for each junction node to their original values stored in the demand_node list, which 
    is assumed to contain the original demand values.
    """
    
def restore_initial_value():
    for x in range(0,len(tank_idx)):
        en.ENsetnodevalue(tank_idx[x], en.EN_TANKLEVEL, tank_init_level[x])
        

def init_demands():
    demand_dict=dict()
    for idx in junction_idx:
        
        max_demand,min_demand=helper.get_demand_limits(idx)
        demand_dict[idx]={"max":max_demand,"min":min_demand}
        #print ('demands max e min: ', max_demand, min_demand)
    return demand_dict

def set_random_demands(demand_dict):
    """Setting random demands for the network
    """
    
    #Iterate demand indexes
    for idx2 in index:
        generate_constant_pattern(1,junction_idx[idx2])
        idx=junction_idx[idx2]
        max_d=demand_dict[idx]["max"]
        min_d=demand_dict [idx]["min"]
        random_demand=uniform(min_d,max_d)
        en.ENsetnodevalue(idx,en.EN_BASEDEMAND,random_demand)

     


def data_m():
    tanks_hour=append_tanks() # dicionario com nivel de cada reservatorio
    
    pumps_st=append_pumps()  # dicionario com estado da bomba 
    # flow=append_flow()
    pumps_kw=pump_kw() # dicionario com potencia de cada bomba 
    demands=demand_value()
    
    t = en.ENrunH()
    time_step = en.ENnextH()
    #time_step=300
    tanks_final_hour=append_final_tanks()

    #Dataframe
    tank_level = pd.DataFrame.from_dict(tanks_hour)
    tank_final= pd.DataFrame.from_dict(tanks_final_hour)
    pump_state= pd.DataFrame.from_dict(pumps_st)
    pump_power= pd.DataFrame.from_dict(pumps_kw)
    demands = pd.DataFrame.from_dict(demands)
    

    step.append(time_step) 
    f = pd.DataFrame(step, columns=['t_step'])

    df = pd.concat([f,tank_level,pump_state, pump_power, demands, tank_final],axis=1)
    
    return df

# %%  Abrir inp
n=30000
df_2 = pd.DataFrame()

for i in range(n):
    err_code = en.ENopen("Bomba-deposito_Sarav2.inp", "report.rpt", " ")

    tank_idx,junction_idx = tank_indexes()

    pump_idx = pump_indexes()
    tanks,tank_init_level,tanks_final=dic_tanks()
    pumps,pump=dic_pumps()
    demand_hour = []
    junction_demand,index,demand_node = junction_base_demand()
    
        

    step= []
    time = []
    
    # Time increment -> hydstep==1 | duration==0
    time_horizon = en.ENgettimeparam(0)
    time_increment = en.ENgettimeparam(1)
    #print("time_horizon", time_horizon)
    #print("time_increment", time_increment)

    # alterar os valores
    en.ENsettimeparam(0, 86400) # Experimentar para 1 mês 86400*30
    en.ENsettimeparam(1, 300) # 5  minutos
    time_horizon2 = en.ENgettimeparam(0)
    time_increment2 = en.ENgettimeparam(1)
    #print("time_horizon alterado", time_horizon2)
    #print("time_increment alterado", time_increment2)

    restore_initial_value()
    restore_demand_value()
    set_initial_level_tank()
    demand_dict = init_demands()
    set_random_demands(demand_dict)
    
    t_step = 1
    en.ENopenH()
    en.ENinitH(10)
   

    set_pump_state() # defino estado da bomba aleatoriamente

    t = en.ENrunH()

    final_2rows= data_m()
    #print('Final 2 rows\n', final_2rows)
    #input()
    
 
    df_2 = pd.concat([df_2, final_2rows], ignore_index=True)  
    
    en.ENcloseH()
    en.ENclose()
    
#print('df_2 \n ', df_2 )



df_output =pd.DataFrame() 
df_output["Δtank_0"] = ((df_2["tank_final_0"]-df_2["tank_0"])/df_2.t_step) #m/s
df_output["pumps_energy"]=(df_2['pumps_kw_0']/3600)*df_2['t_step']
#print('df_output \n ', df_output )
print('DATAFRAME CREATED')
input()

# Saving samples
df_output.to_csv(r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\samples\Fontinha\samples_out_30_000.csv',index=False)
df_2.to_csv(r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\samples\Fontinha\samples_30_000.csv',index=False)
print('saved')


# %% 
# DISTRIBUTION OF THE DATA PLOTS

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


df_plot = pd.read_csv(r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\samples\Fontinha\samples_30_000.csv', index_col=False)
df_out_plot = pd.read_csv(r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\samples\Fontinha\samples_out_30_000.csv', index_col=False)
tank = df_plot.tank_0
demand1 = df_plot.junction_1
demand2 = df_plot.junction_2
tank_var = df_out_plot.Δtank_0



# Create a histogram with a kernel density estimate
save_directory = r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\Samples\Results\Fontinha'
with plt.style.context('science'):
    sns.histplot(data=tank, kde=True)
    plt.title('Distribution of Tank initial level')
    plt.xlabel('Tank level')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_directory,"distribution_tank_level.pdf"), dpi=300)
    plt.show()

    sns.histplot(data=demand1, kde=True)
    plt.title('Distribution of demands in junction1')
    plt.xlabel('Demands')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_directory,"distribution_demands_junction1.pdf"), dpi=300)
    plt.show()

    sns.histplot(data=demand2, kde=True)
    plt.title('Distribution of demands in junction2')
    plt.xlabel('Demands')
    plt.ylabel('Frequency')
    # Display the plots
    plt.savefig(os.path.join(save_directory,"distribution_demands_junction2.pdf"), dpi=300)
    plt.show()

    sns.histplot(data=tank_var, kde=True)
    plt.title('Distribution of tank variation')
    plt.xlabel('Tank Variation')
    plt.ylabel('Frequency')
    # Display the plots
    plt.show()
    plt.savefig(os.path.join(save_directory,"distribution_tank_variation.pdf"), dpi=300)


# %%
