# %% Funções
import epamodule as en
import numpy as np
from random import randint, uniform,choices, seed
import pandas as pd
import matplotlib.pyplot as plt



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
    demand_hour.append(demand)
    return demand_hour


def set_pump_state():  # Definir o estado da bomba aleatoriamente
    for x in range(0, len(pump_idx)):
        # # ciclo if tiago
        for j in range(0, len(tank_idx)):
            # lim = 0.5  # só para varios tanques. Fontinha # esta linha
            # min_tank_level = en.ENgetnodevalue(tank_idx[j], en.EN_MINLEVEL) # só para varios tanques. Fontinha # esta linha
            # max_tank_level = en.ENgetnodevalue(tank_idx[j], en.EN_MAXLEVEL) # só para varios tanques. Fontinha # esta linha
            if len(tanks["tank_"+str(j)]) == 0:
                 en.ENsetlinkvalue(pump_idx[x], en.EN_STATUS, randint(0, 1))
            elif tanks["tank_"+str(j)][-1] < 2: # Fontinha <2 # richmond min_tank_level + lim
                 en.ENsetlinkvalue(pump_idx[x], en.EN_STATUS, 1)  # liga
            elif tanks["tank_"+str(j)][-1] > 7: # fontinha >7 # richmond max_tank_level - lim
                en.ENsetlinkvalue(pump_idx[x], en.EN_STATUS, 0)  # desliga
            else:
                en.ENsetlinkvalue(pump_idx[x], en.EN_STATUS, randint(0, 1))
        
def set_initial_level_tank():
    for m in range(0,len(tank_idx)):
        #value = en.ENgetnodevalue(tank_idx[m], en.EN_TANKLEVEL)
        #lim = 0.3 # fontinha 1# richmond 0.3 
        min_tank_level = en.ENgetnodevalue(tank_idx[m], en.EN_MINLEVEL)
        max_tank_level = en.ENgetnodevalue(tank_idx[m], en.EN_MAXLEVEL)
        en.ENsetnodevalue(tank_idx[m], en.EN_TANKLEVEL, uniform(min_tank_level,max_tank_level)) # value*uniform(0.75, 1.25) novo nivel será maior ou menor 25% que o outro
                                                        # richmond (del value) uniform(min_tank_level + lim, max_tank_level - lim

def set_initial_demand_value():
    for n in index: 
        value = en.ENgetnodevalue(junction_idx[n], en.EN_BASEDEMAND)*uniform(0.75,1.2) # Mesma coisa para os valores de demanda
        en.ENsetnodevalue(junction_idx[n], en.EN_BASEDEMAND,value) 
      
        
        
def demand_value():
    for n in index:
        value = en.ENgetnodevalue(junction_idx[n], en.EN_DEMAND)
        junction_demand["junction_"+str(n)].append(value)
        #print(junction_demand)
    #input()
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
        

# %%  Abrir inp

err_code = en.ENopen("Bomba-deposito_Sarav2_experiencia.inp", "report.rpt", " ")

tank_idx,junction_idx = tank_indexes()

pump_idx = pump_indexes()
tanks,tank_init_level,tanks_final=dic_tanks()
pumps,pump=dic_pumps()
demand_hour = []
junction_demand,index,demand_node = junction_base_demand()

time_step= []
time = []


   
restore_initial_value()
restore_demand_value()
set_initial_level_tank()
set_initial_demand_value()



# Time increment -> hydstep==1 | duration==0
time_horizon = en.ENgettimeparam(0)
time_increment = en.ENgettimeparam(1)
print("time_horizon", time_horizon)
print("time_increment", time_increment)

# alterar os valores
en.ENsettimeparam(0, 86400*30) # Experimentar para 1 mês 86400*30
en.ENsettimeparam(1, 300) # 5  minutos
time_horizon2 = en.ENgettimeparam(0)
time_increment2 = en.ENgettimeparam(1)
print("time_horizon alterado", time_horizon2)
print("time_increment alterado", time_increment2)


t_step = 1
en.ENopenH()
en.ENinitH(10)


while t_step > 0:
    set_pump_state() # defino estado da bomba aleatoriamente
    t = en.ENrunH()
    time_step.append(t_step) 
        
    # Dicionario com o valor de consumo de cada nó
    set_junction=demand_value()
    # dicionario com nivel de cada reservatorio
    tanks_hour=append_tanks()
    # dicionario com estado da bomba 
    pumps_st=append_pumps()
    
    # flow=append_flow()
    # dicionario com energia de cada bomba 
    pumps_kw=pump_kw()
      
    time.append(t)  
    t_step = en.ENnextH()
    tanks_final_hour=append_final_tanks()    
    #time_step.append(t_step)   
    
   
en.ENcloseH()
en.ENclose()
print(sum(time_step))

# %% Definir DataFrame
t = pd.DataFrame(time, columns=['time'])
a = pd.DataFrame.from_dict(tanks_hour)
b = pd.DataFrame.from_dict(tanks_final_hour)
c = pd.DataFrame.from_dict(pumps_st)
d= pd.DataFrame.from_dict(pumps_kw)
e= pd.DataFrame.from_dict(set_junction)
f = pd.DataFrame(time_step, columns=['t_step'])


epanet_data=pd.concat([t,f,a,b,c,d,e],axis=1)


epanet_data.to_csv(r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\Samples\Fontinha\experiencia.csv',index=False)


# %%
teste=t/300
non_integers = teste.loc[~teste['time'].astype(int).eq(teste['time'])]
print(non_integers)
# %%
#import pandas as pd 
#import matplotlib.pyplot as plt

#epanet_data = pd.read_csv(r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Códigos\Test_epanet\Samples\Fontinha\48h_5min_fontinha.csv', index_col=False)

time_hour = epanet_data.time/3600
plt.figure(figsize=(10,6))
plt.figure().set_figwidth(15)

plt.subplot(2,2,1)
plt.plot(time_hour,epanet_data.tank_0, label = "Tank Level")
plt.xlabel('Time [hour]')
plt.legend()
plt.subplot(2,2,2)
plt.plot(time_hour,epanet_data.junction_1, 'r',  label = "Demand Junction1")
plt.legend()
plt.xlabel('Time [hour]')
plt.subplot(2,2,3)
plt.plot(time_hour,epanet_data.pumps_0, 'orange' , label = "Pump State")
plt.xlabel('Time [hour]')
plt.legend()
plt.subplot(2,2,4)
plt.plot(time_hour,epanet_data.junction_2, 'g', label = "Demand Junction2" )
plt.xlabel('Time [hour]')
plt.legend()
plt.suptitle('48h, 5 min ', fontsize=16, fontweight="bold")
plt.show()
# %%
