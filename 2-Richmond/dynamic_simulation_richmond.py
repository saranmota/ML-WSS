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
#seed(42)
def generate_constant_pattern(constant_val,idx):

    my_pattern_index=int(en.ENgetnodevalue(idx,en.EN_PATTERN))
    
    #Get pattern length
    pattern_length=en.ENgetpatternlen(my_pattern_index)

    #new_pattern_name="constant_pattern_"+str(constant_val)+str(idx)
    #ep.ENaddpattern(new_pattern_name)
    #pattern_length=24
    #ep.ENsetpatternlen(new_pattern_name, pattern_length)
    
    #new_pattern_idx=ep.ENgetpatternindex(new_pattern_name)
    for period in range(1, pattern_length + 1):
        en.ENsetpatternvalue(my_pattern_index, period, constant_val)
    #ep.ENsetpatternvalue(new_pattern_idx,pattern_length,constant_val)

    #ep.ENsetnodevalue(idx,ep.EN_PATTERN,new_pattern_idx)

    #Debugging purposes 
    #my_pattern_index=int(ep.ENgetnodevalue(idx,ep.EN_PATTERN))
    #for i in range(1, pattern_length + 1):
    #    pattern_value = en.ENgetpatternvalue(my_pattern_index, i)

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
        my_pump_idx=pump_idx[x]
        pump_val=round(en.ENgetlinkvalue(my_pump_idx,en.EN_STATUS),2)
      
        pumps["pumps_"+str(x)].append(pump_val)
    return pumps   
#
 #def append_flow():
 #    #for x in range(0,len(pump_idx)):
 #    flow["Flow"].append(round(en.ENgetlinkvalue(4,en.EN_FLOW),2))
 #   
 #    return flow


def pump_kw():
    for x in range(0,len(pump_idx)):
       
        pump_energy=round(en.ENgetlinkvalue(pump_idx[x],en.EN_ENERGY),2)
      
        pump["pumps_kw_"+str(x)].append(pump_energy)
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
    for x in pump_idx:
        for j in range(0, len(tank_idx)):
            lim = 0.3  # só para varios tanques. Fontinha # esta linha
            min_tank_level = en.ENgetnodevalue(tank_idx[j], en.EN_MINLEVEL) # só para varios tanques. Fontinha # esta linha
            max_tank_level = en.ENgetnodevalue(tank_idx[j], en.EN_MAXLEVEL) # só para varios tanques. Fontinha # esta linha

            if len(tanks["tank_"+str(j)]) == 0:
                val_set=randint(0, 1) #aleatorio
            elif tanks["tank_"+str(j)][-1] < min_tank_level + lim: 
                val_set=1   # liga
            elif tanks["tank_"+str(j)][-1] > max_tank_level - lim: 
                val_set=0  # desliga
            else:
                val_set=randint(0, 1)
                
           
            en.ENsetlinkvalue(x, en.EN_STATUS, val_set)
        
def set_initial_level_tank():
    for m in range(0,len(tank_idx)):
        min_tank_level = en.ENgetnodevalue(tank_idx[m], en.EN_MINLEVEL)
        max_tank_level = en.ENgetnodevalue(tank_idx[m], en.EN_MAXLEVEL)
        en.ENsetnodevalue(tank_idx[m], en.EN_TANKLEVEL, uniform(min_tank_level, max_tank_level)) # distribuição normal entre minimo e maximo do tanque
                                                        # min e max do tanque aumentados no inp para aumentar range de dados de treino
                                                       

def set_initial_demand_value():
    for n in index: 
        value = en.ENgetnodevalue(junction_idx[n], en.EN_DEMAND)*uniform(0.75,1.2) 
        en.ENsetnodevalue(junction_idx[n], en.EN_DEMAND,value) 
      
        
        
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
    
    dict_limit={}
    for n in index:
        my_index=junction_idx[n]
        max_limit,min_limit=helper.get_demand_limits(my_index)
        dict_limit[my_index]={"max":max_limit,"min":min_limit}    
    return dict_limit

def set_random_demands(demand_dict):
    """Setting random demands for the network
    """
    
    #Iterate demand indexes
    for idx2 in index:
        generate_constant_pattern(1,junction_idx[idx2])
        idx=junction_idx[idx2]
        max_d=demand_dict[idx]["max"] 
        min_d=demand_dict [idx]["min"] 
        random_pattern=uniform(min_d,max_d) # 24/05 Alterei random_demand para random pattern(porque a função estava a ir buscar os pattern values)
        Base_demand = en.ENgetnodevalue(idx,en.EN_BASEDEMAND)
        actual_demand = Base_demand*random_pattern
        en.ENsetnodevalue(idx,en.EN_BASEDEMAND,actual_demand)

     
def data_m():
    tanks_hour=append_tanks() # dicionario com nivel de cada reservatorio
    
    pumps_st=append_pumps()  # dicionario com estado da bomba 
    # flow=append_flow()
    
    pumps_kw=pump_kw() # dicionario com energia de cada bomba 
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
    
    # Debug para verificar quando retornava valores de potencia bomba muito baixos ( situação peculiar richond)
    #my_data=pd.DataFrame()
    #my_dict={}
    #if ((df['pumps_kw_0'] > 0.5) & (df['pumps_kw_0'] < 1)).any():
    #    flow = en.ENgetlinkvalue(45,en.EN_FLOW)
    #    power=en.ENgetlinkvalue(45,en.EN_ENERGY)
    #    presao_junction_afrente = en.ENgetnodevalue(29,en.EN_PRESSURE)
    #    pressao_junction_atras = en.ENgetnodevalue(28,en.EN_PRESSURE)
    #    tank1_idx=en.ENgetnodeindex("F")
    #    print('Tank_idx', tank1_idx)
    #    pressao_tank_afrente = en.ENgetnodevalue(tank1_idx,en.EN_PRESSURE)
    #    tank2_idx=en.ENgetnodeindex("E")
    #    print('Tank2_idx', tank2_idx)
    #    pressao_tank_atras = en.ENgetnodevalue(tank2_idx, en.EN_PRESSURE)
    #    estado = en.ENgetlinkvalue(45,en.EN_STATUS)
    #    demand_frente = en.ENgetnodevalue(29,en.EN_BASEDEMAND)
    #    demand_atras = en.ENgetnodevalue(28,en.EN_BASEDEMAND)
    #    
    #    my_dict["Flow"]=flow
    #    my_dict["Power"]=power
    #    my_dict['pressao_junction_afrente'] = presao_junction_afrente
    #    my_dict['pressao_junction_atras'] = pressao_junction_atras
    #    my_dict['pressao_tank_1'] = pressao_tank_afrente
    #    my_dict['pressao_tank_2'] = pressao_tank_atras  
    #    my_dict['estado']  = estado
    #    my_dict['demand_frente']  = demand_frente
    #    my_dict['demand_atras']  = demand_atras
    #    my_dict["Category"]="Anomaly"
#
    #    en.ENclose()
    #else:
    #    flow = en.ENgetlinkvalue(45,en.EN_FLOW)
    #    power=en.ENgetlinkvalue(45,en.EN_ENERGY)
    #    #curva = en.ENgetcurve(1883)
    #    presao_junction_afrente = en.ENgetnodevalue(29,en.EN_PRESSURE)
    #    pressao_junction_atras = en.ENgetnodevalue(28,en.EN_PRESSURE)
    #    tank1_idx=en.ENgetnodeindex("F")
    #    #print('Tank_idx', tank1_idx)
    #    pressao_tank_afrente = en.ENgetnodevalue(tank1_idx,en.EN_PRESSURE)
    #    tank2_idx=en.ENgetnodeindex("E")
    #    #print('Tank2_idx', tank2_idx)
    #    pressao_tank_atras = en.ENgetnodevalue(tank2_idx, en.EN_PRESSURE)
    #    estado = en.ENgetlinkvalue(45,en.EN_STATUS)
    #    demand_frente = en.ENgetnodevalue(29,en.EN_BASEDEMAND)
    #    demand_atras = en.ENgetnodevalue(28,en.EN_BASEDEMAND)        
    #    
    #    my_dict["Flow"]=flow
    #    my_dict["Power"]=power
    #    my_dict['pressao_junction_afrente'] = presao_junction_afrente
    #    my_dict['pressao_junction_atras'] = pressao_junction_atras
    #    my_dict['pressao_tank_1'] = pressao_tank_afrente
    #    my_dict['pressao_tank_2'] = pressao_tank_atras   
    #    my_dict['estado']  = estado 
    #    my_dict['demand_frente']  = demand_frente
    #    my_dict['demand_atras']  = demand_atras
    #    my_dict["Category"]="Normal"
   #
#
    #    #print('demands atras',demand_atras)
    #    #print('demands frente',demand_frente)
    #    #input()
    #    # Append my_dict to the DataFrame
    #my_data = my_data.append(my_dict, ignore_index=True)
#
    #    # Write the DataFrame to the CSV file
    #my_data.to_csv('rich_samples_test.csv', index=False, mode = 'a')
    return df

# %%  Abrir inp
n=30000
df_2 = pd.DataFrame()
for i in range(n):
    err_code = en.ENopen("Richmond_deleted_rules.inp", "report.rpt", " ")

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


    # alterar os valores
    en.ENsettimeparam(0, 86400) # Experimentar para 1 mês 86400*30
    en.ENsettimeparam(1, 300) # 5  minutos
    time_horizon2 = en.ENgettimeparam(0)
    time_increment2 = en.ENgettimeparam(1)


    restore_initial_value()
    restore_demand_value()
    set_initial_level_tank()
    demand_dict = init_demands()
    set_random_demands(demand_dict)
    
    t_step = 1
    en.ENopenH()
    en.ENinitH(10)
   
   # set pump state antes do run 
    set_pump_state() # defino estado da bomba aleatoriamente

    t = en.ENrunH()

    final_2rows= data_m()
    
    df_2 = pd.concat([df_2, final_2rows], ignore_index=True)  
    
    en.ENcloseH()
    en.ENclose()
    
#print('df_2 \n ', df_2 )
#input()

df_output =pd.DataFrame() 

for k in range(0,6):
    df_output["Δtank_"+str(k)] = ((df_2["tank_final_"+str(k)]-df_2["tank_"+str(k)])/df_2.t_step) #m/s

print('DATAFRAME CREATED')
input()

# Saving samples
df_output.to_csv(r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\samples\Richmond\rich_dynamic_out_30_000.csv',index=False)
df_2.to_csv(r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\samples\Richmond\rich_dynamic_30_000.csv',index=False)
print('saved')

exit()

# %% 
# DISTRIBUTION OF THE DATA PLOTS

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import scienceplots
import os 
import re


df_plot = pd.read_csv(r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\samples\Richmond\rich_dynamic_30_000.csv', index_col=False)
df_out_plot = pd.read_csv(r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\samples\Richmond\rich_dynamic_out_30_000.csv', index_col=False)
#tanks = df_plot.tank_0
#demands = df_plot.junction_1
#demand2 = df_plot.junction_2
#tanks_var = df_out_plot.Δtank_0
save_directory = r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\Samples\Results\Richmond'

# Iterate over each tank column in the dataframe
with plt.style.context('science'):
    for column in df_plot.columns:
        if column.startswith('tank_'):
            plt.figure()
            # Create a histogram with kernel density estimate for the current tank
            sns.histplot(data=df_plot[column], kde=True)
            plt.title(f'Distribution of {column} initial level')
            plt.xlabel('Tank level')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(save_directory, "distribution_" + column + ".pdf"), dpi=300)

# Iterate over each demand column in the dataframe
with plt.style.context('science'):
    for column in df_plot.columns:
        if column.startswith('junction_'):
            plt.figure()
            # Create a histogram with kernel density estimate for the current tank
            sns.histplot(data=df_plot[column], kde=True)
            plt.title(f'Distribution of {column} demand')
            plt.xlabel('Demand')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(save_directory, "distribution_" + column + ".pdf"), dpi=300)
        
# Iterate over each tankΔ column in the dataframe
with plt.style.context('science'):
    for column in df_out_plot.columns:
        if column.startswith('Δtank_'):
            plt.figure()
            # Create a histogram with kernel density estimate for the current tank
            sns.histplot(data=df_out_plot[column], kde=True)
            plt.title(f'Distribution of {column} level')
            plt.xlabel('Variation')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(save_directory, "distribution_" + column + ".pdf"), dpi=300)
# %%
