
import epamodule as ep
from datetime import datetime,timedelta
import time
import sys
import random
import pytz
import ctypes
#Time relatable functions

#TODO separate this into two files
#1- actual typical helper functions
#2- object oriented interaction with any epanet module


def get_timestampz(my_time):

    f = '%Y-%m-%d %H:%M:%S'
    ts=time.gmtime(my_time)
    time_stamp=time.strftime(f, ts)
    date_time_obj=datetime.strptime(time_stamp,'%Y-%m-%d %H:%M:%S')
    return date_time_obj



def close_to_midnight(current_ts,time_step):
    """Calculate if the current time is close to midnight

    Args:
        current_ts (datetime object): current timestamp
        time_step (integer): time step to consider

    Returns:
        boolean: flag indicating if is in fact close enough to midnight
    """


    time_step=time_step//2
    time_step+=50

    today_midnight=current_ts.replace(hour=0,minute=0,second=0,microsecond=0)
    tomorrow_midnight=today_midnight+timedelta(days=1)
    time_to_tomorrow=(tomorrow_midnight-current_ts).total_seconds()
    time_to_yesterday=(current_ts-today_midnight).total_seconds()

    if time_to_tomorrow<time_step or time_to_yesterday<time_step:
        return True
    else:
        return False


#def disable_all_controls():
#
#
#    list_controls=get_all_controls()
#    len_control=len(list_controls)
#
#    for i in range(1,len_control+1):
#        print(ep.ENgetcontrol(i))
#        input()



def calculate_tank_tolerances():


    idx=get_el_idx(ep.EN_NODECOUNT,ep.EN_TANK)[0]

    my_dict={}

    for i in idx:
        
        min_head=ep.ENgetnodevalue(i,ep.EN_MINLEVEL)
        max_head=ep.ENgetnodevalue(i,ep.EN_MAXLEVEL)
        elevation=ep.ENgetnodevalue(i,ep.EN_ELEVATION)
        
        min_elevation=min_head+elevation
        max_elvation=max_head+elevation

        tolerance=2 #in meters

        min_permited=min_elevation+tolerance
        max_permited=max_elvation-tolerance

        my_dict[i]=[min_permited,max_permited]
        
    return my_dict

def set_controls(idx_list,flag_which):

    for idx in idx_list:

        if flag_which=="max":
            node_status=0
        elif flag_which=="min":
            node_status=1
        else:
            node_status=random.choices([0,1])[0]
        
        ep.ENsetlinkvalue(idx,ep.EN_STATUS,node_status)
    

def get_demand_limits(my_idx):
    
    pattern_idx=int(ep.ENgetnodevalue(my_idx,ep.EN_PATTERN))
    pattern_len=ep.ENgetpatternlen(pattern_idx)
    pattern_values=[]
    for i in range(1,pattern_len):
        pattern_values.append(ep.ENgetpatternvalue(pattern_idx,i))
    
    max_demand=max(pattern_values)
    min_demand=min(pattern_values)
    return max_demand,min_demand
    



def get_all_rules():

    i=1
    my_list=[]
    flag_meh=False
    while True:
        my_rules=get_rule(i)

        if my_rules==None:
            break
        else:
            my_list.append(my_rules)

        i+=1
    return my_list



def get_all_controls():
    """Get all types of controls in project

    Returns:
        tuple: list of all controls
    """

    list_controls=[]
    for i in range(1,sys.maxsize):
        new_control=ep.ENgetcontrol(i)
        if new_control==None:
            break
        else:
            list_controls.append(new_control)

    return list_controls


def get_rule(index):
    return ep.ENgetcontrol(index)

def get_pump_idx():
    return get_el_idx(ep.EN_LINKCOUNT,ep.EN_PUMP)[0]

def get_tank_idx():
    return get_el_idx(ep.EN_NODECOUNT,ep.EN_TANK)[0]


def get_el_idx(count_element,type_element):
    """
    count_element: code component from the epamodule wrapper,  
        EN_NODECOUNT
        EN_TANKCOUNT
        EN_LINKCOUNT
        EN_PATCOUNT
        EN_CURVECOUNT
        EN_CONTROLCOUNT
    
    type_element: type code from the epamodule element. 
    For nodes:
        EN_JUNCTION
        EN_RESERVOIR
        EN_TANK

    For links:
        EN_CVPIPE
        EN_PIPE
        EN_PUMP
        EN_PRV 
        EN_PSV
        EN_PBV
        EN_FCV
        EN_TCV
        EN_GPV 

    #WARNING: For now, only nodes and links are implemented
    """


    n_el=ep.ENgetcount(count_element)
    el_idx=[]
    el_id=[]

    for i in range(1,n_el+1):

        if count_element==ep.EN_NODECOUNT:
            some_id=ep.ENgetnodeid(i).decode("UTF-8")
            el_type=ep.ENgetnodetype(i)
        elif count_element==ep.EN_LINKCOUNT:
            el_type=ep.ENgetlinktype(i)
            some_id=ep.ENgetlinkid(i).decode("UTF-8")

        if el_type==type_element:
            el_idx.append(i)
            el_id.append(some_id)
            #Get also the corresponding id
            
    return el_idx,el_id

def get_value(obj_code,my_param,idx):

    temp_vals={}
    #TODO i don't want if for every time i use this param 
    #Find a way
    for i in idx:
        if obj_code==ep.EN_NODECOUNT:
            
            temp_vals[i]=ep.ENgetnodevalue(i,my_param)
            
        elif obj_code==ep.EN_LINKCOUNT:
            temp_vals[i]=ep.ENgetlinkvalue(i,my_param)
        else: 
            print("WARNING: current object code not implemented")
    return temp_vals


def set_value(obj_code,my_param,idx,value):

    #NOTE going to assume that idx is a list

    #for i in idx:
    if obj_code==ep.EN_NODECOUNT:
        ep.ENsetnodevalue(idx,my_param,value)
    elif obj_code==ep.EN_LINKCOUNT:
        #print(type(idx),type(my_param),type(value))
        ep.ENsetlinkvalue(idx,my_param,value)




    
def extract_dict(main_dict,idx):
    new_dict={}
    for my_key in main_dict:
        new_dict[my_key]=main_dict[my_key][idx]
    return new_dict




def get_tank_levels(tanks_idx,ep_code):
    temp_tank_levels={}
    for i in tanks_idx:
        temp_tank_levels[i]=ep.ENgetnodevalue(i,ep_code)
    return temp_tank_levels


def get_pump_values(pumps_idx):

    pump_status={}
    pump_energy={}

    for i in pumps_idx:

        pump_status[i]=ep.ENgetlinkvalue(i,ep.EN_STATUS)
        pump_energy[i]=ep.ENgetlinkvalue(i,ep.EN_ENERGY)

    return pump_status,pump_energy 


def get_demand_idx():
    n_el=ep.ENgetcount(ep.EN_NODECOUNT)
    demand_idx=[]
    demand_id=[]
    for i in range(1,n_el+1):
        if (ep.ENgetnodetype(i)==ep.EN_JUNCTION) and (ep.ENgetnodevalue(i,ep.EN_BASEDEMAND)>0):
            demand_id.append(ep.ENgetnodeid(i).decode("UTF-8"))
            demand_idx.append(i)
    return demand_idx,demand_id

def get_demand_values(demand_idx):

    current_demand_value={}
    for i in demand_idx:
        #TODO correct this, i'm just testing stuff
        #demand_value=ep.ENgetnodevalue(i,ep.EN_BASEDEMAND)
        demand_value=ep.ENgetnodevalue(i,ep.EN_DEMAND)
        current_demand_value[i]=demand_value

    return current_demand_value

def convert_idx_id(some_dict,type_obj="node"):

    new_dict={}
    for my_key in some_dict:
        #Calculate corresponding id
        if type_obj=="node":
            my_id=ep.ENgetnodeid(my_key).decode("UTF-8")
        elif type_obj=="link":
            my_id=ep.ENgetlinkid(my_key).decode("UTF-8")

        new_dict[my_id]=some_dict[my_key]
    return new_dict




#################################################
####A place for super simple helper functions####
#################################################

def datetime2string(a_date):
    f="%m/%d/%Y, %H:%M:%S"
    my_string = a_date.strftime(f)
    return my_string


def t_data_length(time_hours,time_step_seconds):
    #Convert hours to seconds
    sec_lenghts=time_hours*60*60
    n_size=int(sec_lenghts/time_step_seconds)
    return n_size



def reset_dict_array(a_dict):
    #Make a dictionary in which its values are array with an initial value
    
    new_dict={}
    for my_key in a_dict:
        new_dict[my_key]=[a_dict[my_key]]

    return new_dict

def append_dict2array(dict_array,a_dict):

    for my_key in a_dict:
        dict_array[my_key].append(a_dict[my_key])
    return dict_array

def day2sec(n_days):

    return n_days*24*60*60



#def make_init_DA(a_dict):
#
#    dict_array={}
#
#    for my_key in a_dict:
#        dict_array[my_key]=[a_dict[my_key]]
#
#    return dict_array

    


##################################################
####Some Thrash that i don't want to erase yet####
##################################################

#def get_init_demand_idx():
#
#    n_el=ep.ENgetcount(ep.EN_NODECOUNT)
#    demand_idx={}
#    for i in range(1,n_el+1):
#
#        if ep.ENgetnodetype(i)==ep.EN_JUNCTION:
#            if ep.ENgetnodevalue(i,ep.EN_BASEDEMAND)>0:
#                demand_idx[i]=[ep.ENgetnodevalue(i,ep.EN_BASEDEMAND)]
#
#    return demand_idx


#def tanks_levels(tanks_idx):

#

#    current_level={}

#    for i in tanks_idx:

#        level_value=ep.ENgetnodevalue(i,ep.EN_TANKLEVEL)

#        current_level[i]=level_value

#    return current_level
