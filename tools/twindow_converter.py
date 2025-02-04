
from typing import List, Tuple, Optional
import copy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
#from utils_conv import *
import random 
#from converter_core import converter_core

random.seed(41)
class twindow_converter:

    def __init__(self,window_size: int,n_windows: int,time_step: int,time_horizon: int,n_pumps: int):
        

        #Generate
        self.window_size=window_size
        self.n_windows=n_windows
        self.time_step=time_step
        self.time_horizon=time_horizon
        self.n_pumps=n_pumps
        
        self.cur_time=0
        #self.cur_time=int(datetime.datetime.now().timestamp())
        #elf.n_list=int(self.time_horizon/float(self.duty_cycle))


    def single_converter(self,pos_array: List[int],window_array: List[int])->Tuple[List[float],List[int]]: # pos_array = position array(time)

        #Define initial time
        pump_status=[]
        time_array=[]

        if pos_array[0]!=self.cur_time:  # Se o primeiro elemento do pos_array for diferente do tempo atual
            time_array.append(self.cur_time)   # Se sim, , adiciona o tempo atual à lista time_array
            pump_status.append(0)              # e o valor 0 ao pump_status

        i=0
        while i < len(pos_array): 
            
            #Add time
            time_array.append(pos_array[i])
            pump_status.append(1)

            while True:
                if i!=len(pos_array):

                    if i!=len(pos_array)-1:

                        if pos_array[i]+window_array[i]==pos_array[i+1]:# verifica se window atual terminou na proxima window
                            i+=1
                    
                        else:
                            time_array.append(pos_array[i]+window_array[i]) # Se não bomba desligada.
                            pump_status.append(0)
                            i+=1
                            break
                    else:
                        time_array.append(pos_array[i]+window_array[i])
                        pump_status.append(0)
                        i+=1
                        break

                else:
                    break

        time_array.append(self.cur_time+self.time_horizon)
        pump_status.append(0)
        
        return pump_status,time_array



    def generate_continuous_data(self,type_generator: Optional[str]="Full")->Tuple[List[int],List[int]]:
        """ Generation of random data"""

        print("Generating data")
        #
        min_range=self.cur_time
        max_range=min_range+self.time_horizon

        pos_list=[]
        window_list=[]

        for i in range(self.n_windows):

            #Verify if window is appropriate
            flag_verified=False
            while not flag_verified:
                #Generate position in time, and length
                pos=random.randint(min_range,max_range)
                length_win=random.randint(1,self.window_size)

                #Check if in bounds
                if pos+length_win<max_range:
                  
                    #Check if 
                    if len(pos_list)==0:
                        pos_list.append(pos)
                        window_list.append(length_win)
                        flag_verified=True
                    else:
                        flag_continue=True
                        for i in range(len(pos_list)):
                            pos_i=pos_list[i]
                            length_i=window_list[i]

                            #NOTE going to substract from the back and from the front 
                            if (pos_i<pos and pos_i+length_i>pos+length_win) or (pos<pos_i and pos+length_win>pos_i+length_i) or (pos<pos_i+length_i and pos>pos_i) or (pos+length_win>pos_i and pos+length_win<pos_i+length_i):
                                flag_continue=False
                                break
                        

                        if flag_continue:

                            pos_list.append(pos)
                            window_list.append(length_win)
                            flag_verified=True


        #Sort the list
        # Sort list A and get the corresponding indices of the sorted values
        sorted_indices = sorted(range(len(pos_list)), key=lambda k: pos_list[k])
        pos_list = [pos_list[i] for i in sorted_indices]
        window_list = [window_list[i] for i in sorted_indices]

        assert len(pos_list)==len(window_list) #Small verification

        return pos_list,window_list
    

    def multi_converter(self,full_pos: List[int])->Tuple[List[int],List[int]]:#Tuple[List[List[float]]]:
        """
            Convert several windowed punp schedules to discrete format
        """

        main_time=[]
        main_disc_control=[]
        final_control_array=[]

        size_list=len(full_pos)

        pos_list=full_pos[0:int(size_list/2)]
        window_list=full_pos[int(size_list/2):]
        
        final_pos_list=[]
        final_window_list=[]
        #


        #Raw list to several lists
        for i in range(self.n_pumps):
            new_pos_list=pos_list[i*self.n_windows:(i+1)*self.n_windows]
            new_windows_list=window_list[i*self.n_windows:(i+1)*self.n_windows]

            final_pos_list.append(new_pos_list)
            final_window_list.append(new_windows_list)


        #Individual conversion of each pump schedule to discrete format
        for idx in range(len(final_pos_list)): 

            dis_control, new_control_time=self.single_converter(final_pos_list[idx],final_window_list[idx])
            
            #self.visualizer_disc(dis_control,new_control_time)
            #Turn it into a integer
            new_control_time=[int(x) for x in new_control_time]
            main_disc_control.append(dis_control)
            main_time.append(new_control_time)
        
    

        main_time_copy=copy.deepcopy(main_time)
        #Extend main_time by 1 more value, to indicate end of control
        for i in range(self.n_pumps):
            #Get last time of each pump
            last_time=main_time_copy[i][-1]
            #time_to_end=self.time_horizon-last_time
            main_time_copy[i].append(self.time_horizon+last_time)


        pump_idx=[0 for _ in range(self.n_pumps)]
        

        cur_time=main_time_copy[0][0]
        last_time=cur_time+self.time_horizon
        final_time_array=[cur_time]

        cur_pump_status=[[pumps[0]] for pumps in main_disc_control]

        while cur_time<last_time:
            #print("Debugging loop")
            #print("Current time: ",cur_time)
            #print("last time: ",last_time)
            #print("Current pump status: ",cur_pump_status)
            
            current_time_array=[]

            #Find the pump with the lowest time gap
            #print("Starting loop \n")

            for i in range(self.n_pumps):
              
                new_time=main_time_copy[i][pump_idx[i]+1]
                time_dif=new_time-(cur_time)
  
                if time_dif>=self.time_step:
                    time_dif=self.time_step
                current_time_array.append(time_dif)

            #In current_time_array, find lowest value and its index
            min_left_time=min(current_time_array)
            
            if min_left_time==0:
                #A minimum left time means that a duty cycle ended without changing the pump status
                for idx in range(len(current_time_array)):
                    if current_time_array[idx]==min_left_time:
                        pump_idx[idx]+=1
                continue


            #Find all indices with the same value
            same_time=[0 for _ in range(self.n_pumps)]
            for i in range(self.n_pumps):
                if current_time_array[i]==min_left_time:
                    same_time[i]=1
                else:
                    same_time[i]=0

            #From those with the same min_time, check which ones has a change of pump status
            switch_pump=[0 for _ in range(self.n_pumps)]
            for i in range(self.n_pumps):
                if same_time[i]==0:
                    pass
                else:
                    #Check if pump is actually going to cheange 
                    #Check if next time step coincides with the min_left
                    pump_left_time=main_time_copy[i][pump_idx[i]+1]
                    #Check if there are any pump status 
                    if pump_idx[i]==len(main_disc_control[i])-1:
                        switch_pump[i]=0
                    else:

                        if pump_left_time==cur_time+min_left_time and cur_pump_status[i][-1]!=main_disc_control[i][pump_idx[i]+1]:
                            switch_pump[i]=1

            #Update time array and current_time
            cur_time=cur_time+min_left_time

            
            final_time_array.append(cur_time)

            for i in range(self.n_pumps):
                if switch_pump[i]==1:
                    pump_idx[i]=pump_idx[i]+1
     
                next_pump_status=main_disc_control[i][pump_idx[i]]

                cur_pump_status[i].append(next_pump_status)
            

        return cur_pump_status,final_time_array

    def visualizer_multi(self,all_disc_control: List[List[int]],time_control: List[int])->None:

        many_lists=len(all_disc_control)

        fig, my_axes = plt.subplots(many_lists, 1, sharex=True, figsize=(10, 6))

        fig.text(0.5, 0.04, 'Time', ha='center', va='center')
        fig.text(0.06, 0.5, 'Control Status', ha='center', va='center', rotation='vertical')
        for i in range(many_lists):

                #Plot
            my_axes[i].step(time_control,all_disc_control[i],where="post", linestyle='-', marker='o')
            my_axes[i].fill_between(time_control, all_disc_control[i], alpha=0.8,step="post")
            #my_axes[i].set_xlabel('Time')
            #my_axes[i].set_ylabel('Control Status')
            #my_axes[i].set_title('Continuous Control Status')
            #add as many xticks as control_time, but no xtick values 
            #plt.xticks(range(len(control_time)),control_time)
            
        plt.savefig("./digital_twin_module/converters/some_images/from_window_scheduler_Control_status.png")
        plt.show()



    def visualizer_disc(self,control_array: List[float], control_time: List[int])->None:
        """
        Visualize the control array and control time
        """


        #Create duty_cycle array

        #print("Control array: ",control_array)
        #print("Control time: ",control_time)
        #print("Control_time[0]: ",control_time[0])
        #print("Control_time[-1]: ",control_time[-1])
        #print("Sort control time: ",sorted(control_time))

        plt.figure(figsize=(20,10))
        #plt.bar(range(len(control_time)),control_array,width=bar_width*0.8,facecolor ='blue',edgecolor='black')
        plt.step(control_time,control_array,where="post",linestyle='-', marker='o')
        plt.fill_between(control_time, control_array, alpha=0.8,step="post")
        plt.xlabel('Time')
        plt.xlim(control_time[0],control_time[-1])
        plt.ylabel('Control Status')
        plt.title('Continuous Control Status')
        #add as many xticks as control_time, but no xtick values 



        plt.show()  