import epamodule as ep
import numpy as np
from random import randint, uniform,choices, seed
import pandas as pd
import matplotlib.pyplot as plt
import tools.helper as hp



def init_epanet_v2(self):     
        ep.ENopenH()
        ep.ENinitH(0)
        ep.ENrunH()


def init_demands(self):

    self.demand_dict={}
    idx_demands=self.get_obj_idx("Demand_node")

    for idx in idx_demands:
        max_demand,min_demand=hp.get_demand_limits(idx)
        self.demand_dict[idx]={"max":max_demand,"min":min_demand}


def generate_constant_pattern(constant_val,idx):

    my_pattern_index=int(ep.ENgetnodevalue(idx,ep.EN_PATTERN))
    
    #Get pattern length
    pattern_length=ep.ENgetpatternlen(my_pattern_index)

    #new_pattern_name="constant_pattern_"+str(constant_val)+str(idx)
    #ep.ENaddpattern(new_pattern_name)
    #pattern_length=24
    #ep.ENsetpatternlen(new_pattern_name, pattern_length)
    
    #new_pattern_idx=ep.ENgetpatternindex(new_pattern_name)
    for period in range(1, pattern_length + 1):
        ep.ENsetpatternvalue(my_pattern_index, period, constant_val)
    #ep.ENsetpatternvalue(new_pattern_idx,pattern_length,constant_val)

    #ep.ENsetnodevalue(idx,ep.EN_PATTERN,new_pattern_idx)

    #Debugging purposes 
    #my_pattern_index=int(ep.ENgetnodevalue(idx,ep.EN_PATTERN))
    for i in range(1, pattern_length + 1):
        pattern_value = ep.ENgetpatternvalue(my_pattern_index, i)


def get_tank_tolerances():
    """Get the minimum and maximum head for each tank

    Returns:
        tank_tolerances (dictionary): dictionary containing the minimum and maximum head for each tank
    """
    tank_tolerances={}

    idx_tank=hp.get_tank_idx()

    for idx in idx_tank:

        tank_elevation=ep.ENgetnodevalue(idx,ep.EN_ELEVATION)
        min_level=ep.ENgetnodevalue(idx,ep.EN_MINLEVEL)
        max_level=ep.ENgetnodevalue(idx,ep.EN_MAXLEVEL)
        min_head=min_level+tank_elevation
        max_head=max_level+tank_elevation

        #Impose limitations
        limit=2
        local_tolerances={"min_limit":min_head+limit,"max_limit":max_head-limit,"elevation":tank_elevation}
        tank_tolerances[idx]=local_tolerances
    return tank_tolerances


def update_epanet(self):
    """Update epanet simulator
    """

    _=ep.ENrunH()
    t_step=ep.ENnextH()
    return t_step



def set_random_demands(self,random_only=False,flag_debug=False):
    """Setting random demands for the network
    """
    
    idx_demands=self.get_obj_idx("Demand_node")
    
    for idx in idx_demands:

        #Generate a random pattern
        if not random_only:
            generate_constant_pattern(1,idx)

        #get max and min
        max_d=self.demand_dict[idx]["max"]
        min_d=self.demand_dict[idx]["min"]
        random_demand=random.uniform(min_d,max_d)
        #Debugging
        a_pattern=int(ep.ENgetnodevalue(idx,ep.EN_PATTERN))


        ep.ENsetnodevalue(idx,ep.EN_BASEDEMAND,random_demand)
        self.init_epanet_v2()
        if flag_debug:
            print("a pattern ", a_pattern)
            pattern_value = ep.ENgetpatternvalue(a_pattern, 1)
            #print("pattern value ", pattern_value)



def dynamic_simulation(self,sim_mode="real_time",next_checkpoint=0,flag_save_csv=False,flag_save_db=False,academic=False,when="midnight"):
        """Simulation of the water supply system

        Args:
            current_time (integer): current time in seconds
            hydr_file (str, optional): name of the file of the hydraulic state of the network. Defaults to "".
            sim_mode (str, optional): indicates wether there will be stops like in a real environment. Defaults to "real_time".
            next_checkpoint (int, optional): time until this portion of the simulation stops. Defaults to 0.
            flag_save_csv (bool, optional): if data will be saved to a csv file. Defaults to False.
            flag_save_db (bool, optional): if data will be saved the db. Defaults to False.

        Returns:
            current_time (integer): time in seconds after the end of this simulation
        """


        #Generate constant 1 demand pattern
        
        
        current_time=self.th.now_time(update_time=True)
        if when=="midnight":
            current_time=self.th.get_midnight(current_time)

        
        #Check if simulation will run for a certain time or to infinity
        if next_checkpoint==0:
            next_checkpoint=sys.maxsize

        #NOTE I could perhaps make it even shorter
        old_current_time=current_time

        j=0
        n_samples=20
        while j<n_samples:
            next_checkpoint=current_time+24*60*60*20
            j+=1
            print(j)
            input_file=self.input_file


            #NOTE IMPORTANTE
            self.open_epanet(input_file)

            #Setting up parametrs 
            self.set_random_demands()
            self.set_random_water_levels()
            
            self.init_epanet_v2()
            self.set_controls()
            #Create new file and use it 
            #throwaway_file="throwaway_file.inp"
            #self.restart_epanet(throwaway_file)
            #print("Her")
            ##self.set_random_demands(random_only=True,flag_debug=True)
            #print("Initializing simulation")
            #input()
            ##random water level
           

            #random operations
            

            #Setting system
            #self.init_epanet()

            
            #First iteration
            self.update_system()
            self.data_management_v2(flag_save_csv,self.flag_connect_db,current_time,flag_energy=False)

            t_step=self.update_epanet()
            self.update_system()
            self.data_management_v2(flag_save_csv,self.flag_connect_db,current_time,flag_energy=True)
            
            flag_continue=False
            #t_step=1
            temp_step=t_step
            flag_terminate=False

            while True:



                
                if temp_step==self.time_hydstep:
                    #A time step is completed

                    current_time+=temp_step
                    temp_step=0
                    if current_time>=next_checkpoint and next_checkpoint!=-1:
                        #ep.ENsaveinpfile("temp.inp")
                        current_time+=self.time_hydstep

                        print("here")
                        input()
                        break
                    #else:
                        #NOTE In this matter the initial values are not 
                        #being considered
                        #self.data_management(flag_save_csv,self.flag_connect_db,old_current_time)

                    if academic:
                        #self.set_controls()
                        self.set_controls()
                        self.update_system()
                        self.data_management_v2(flag_save_csv,self.flag_connect_db,current_time,flag_energy=False)
                        flag_continue=True
                        
                        
                        

                #Update systems
                t_step=self.update_epanet()

                if flag_continue:
                    self.update_system()
                    self.data_management_v2(flag_save_csv,self.flag_connect_db,current_time,flag_energy=True)
                    flag_continue=False
                    


                #t_step==0 indicates the simulation has ended 
                if t_step==0:
                    current_time+=self.time_hydstep
                    print("here")
                    input()
                    break
                else: 
                    temp_step+=t_step

            self.close_epanet()