import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from scipy.optimize import BFGS
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
import scienceplots
import os 
import re
import joblib
import converter_v2
import converter_v3


def Cost(x,*args):
    nDutyCycles, timeHorizon = args
    
    tempo_duty, level_duty, potencia_dc, pumpState_dc, tarif_dc, tempo, tarifa, tank_final, level = converter_v2.simulation(x, nDutyCycles, timeHorizon)
    #potencia_inventada = [0.0, 125.0, 0.0, 125.0, 0.0, 125.0, 0.0, 125.0, 0.0, 125.0, 0.00, 125.0, 0.0, 0.0]
    # tempo_duty  está em segundos

    CostT = 0;       
    for i in range(len(tempo_duty)-1):
        dur1 = (tempo_duty[i+1] - tempo_duty[i]) / 3600
        tariffpriceInc = dur1 * tarif_dc[i][0]
        Cost = tariffpriceInc * potencia_dc[i]
        CostT += Cost

    return CostT

def g1(x,nDutyCycles,timeHorizon): #Lower and Higher Water Level 
    
    # Print the input parameters
    print(f"g1 called with x: {x}, nDutyCycles: {nDutyCycles}, timeHorizon: {timeHorizon}")


    tempo_duty, level_duty, potencia_dc, pumpState_dc, tarif_dc, tempo, tarifa, tank_final,level = converter_v2.simulation(x,nDutyCycles,timeHorizon)
    # gi = level_duty
    gi= level
    #print(gi)
    print(f"g1 output: {gi}")
    
    return gi # retorna alturas no inicio e no final do DC e nas 24h


def g2(x): # restrição tempos entre DC's
    # print('Temporal Logic Const. --> x(start-stop)')
    n_arranques=int(len(x)/2)
    g2_F3=[0 for i in range(0,n_arranques)]
    for i in range(0,int(n_arranques-1)):
        # g2_F3[i]=x[i+1]-(x[i]+x[i+n_arranques])
        g2_F3[i]=(x[i]+x[i+n_arranques])-x[i+1]
    g2_F3[n_arranques-1]=(x[n_arranques-1]+x[int(2*n_arranques-1)])-24 # garantir que a ultima duração não é superior a T
    return g2_F3

def eps_definition_F3(x):
    # V3
    epsF_i=0.01
    epsF_d=0.018
    eps_aux=np.zeros(len(x))
    n_dc=int(len(x)/2)

    # progressivas para início --> correções de perturbações com o dc seguinte    
    # INÍCIO DC
    for i in range(0,int(len(x)/2)):
        flagR_i=0            
        inicio=x[i]
        dur=x[i+n_dc]

        if(i!=(int(len(x)/2))-1): 
            next=x[i+1] 
        else:
            next=24

        if(inicio + (max(inicio,1)*epsF_i) + dur < next):
            eps_aux[i]=max(inicio,1)*epsF_i
        else:
            dif=next-inicio-dur
            if(dif>=3e-4):
                eps_aux[i]=dif
            else:
                flagR_i=1                  
                eps_aux[i]=-max(inicio,1)*epsF_i

        if(flagR_i==1): #dif. regressiva para início --> correções de perturbações com o dc anterior
            if(i!=0):
                pre=x[i-1]+x[i-1+n_dc] #final do DC anterior
            else:
                pre=0
                
            if(inicio - (max(inicio,1)*epsF_i) < pre):
                dif=(inicio - pre) 
                if(dif>=3e-4):
                    eps_aux[i]=-dif
                else:
                    eps_aux[i]=max(inicio,1)*epsF_i # vamos sobrepor para a frente...           

    #DURAÇÃO DC
    for j in range(int(len(x)/2),len(x)): 
        inicio=x[j-n_dc]
        dur=x[j]
        flagR_d=0 
        
        if(j!=(len(x)-1)): 
            next=x[j+1-n_dc]
        else:
            next=24
        
        if(dur + (max(dur,1)*epsF_d) + inicio < next):
            eps_aux[j]=max(dur,1)*epsF_d
        else:
            dif=next - dur - inicio
            if(dif>=3e-4):
                eps_aux[j]=dif
            else:
                flagR_d=1                  
                eps_aux[j]=-max(dur,1)*epsF_d

        if(flagR_d==1):         
            if(dur - max(dur,1)*epsF_d <= 0): #acontece para dur==0
                if(dur - 3e-4 >=0):
                    eps_aux[j]=-3e-4

    #eps_aux=np.reshape(eps_aux,(1,int(len(x))))[0] 

    tot=x+eps_aux
    if any(t < 3e-4 for t in tot): 
        print (x)                                
    
    return eps_aux

def diff_Cost(x, *args):
    eps_aux=eps_definition_F3(x)    
    #print('eps_aux', eps_aux)
    #print('x\n ',x)
    jac=approx_fprime(x, Cost, eps_aux, *args) # jac=approx_fprime(x, Cost, eps_aux,*(chart,inp))
    #print('Jac', jac)
    #input()
    return jac

def diff_g1(x,nDutyCycles, timeHorizon):
    eps_aux=eps_definition_F3(x)  
    print(g1(x,nDutyCycles, timeHorizon))
    jac=approx_fprime(x, g1,eps_aux,*(nDutyCycles, timeHorizon), )
    print('Jacobian g1 - Levels and duration \n', jac)
    #print(g1(x,nDutyCycles, timeHorizon))
    #input()
    
    #jac=approx_fprime(x, g1(x,nDutyCycles, timeHorizon), eps_aux)
    return jac

def diff_g2(x):
    eps_aux=eps_definition_F3(x)                    
    jac=approx_fprime(x, g2, eps_aux)
    return jac

def optimization(x0,nIncOpt,nDutyCycles,timeHorizon): 
    #C1 = NonlinearConstraint(lambda x0:  g1(x0,nDutyCycles,timeHorizon),2,7, jac="2-point", keep_feasible=False) #Water Level
    #C1 = NonlinearConstraint(lambda x0:  g1(x0,nDutyCycles,timeHorizon),2,7, jac=lambda x0:diff_g1(x0,nDutyCycles, timeHorizon), keep_feasible=False) #Water Level
    C1 = NonlinearConstraint(lambda x0: g1(x0,nDutyCycles,timeHorizon),2,7, jac=lambda x0:diff_g1(x0,nDutyCycles, timeHorizon), keep_feasible=False) #Water Level

    #C_DC =NonlinearConstraint(lambda x0: g2(x0), -np.inf, -4e-4, hess=BFGS(), jac="2-point", keep_feasible=True) #deltaT entre DC's tem de ser superior a 2 seg
    C_DC =NonlinearConstraint(lambda x0: g2(x0), -np.inf, -4e-4, hess=BFGS(), jac=diff_g2, keep_feasible=True) #deltaT entre DC's tem de ser superior a 2 seg
    bounds=Bounds([0 for i in range(0,nIncOpt)],[24 for i in range(0,nIncOpt)], keep_feasible=True); #minimo de 3 seg
    #bounds=Bounds([0 for i in range(0,288)],[24 for i in range(0,288)], keep_feasible=True); #minimo de 3 seg

    #res=minimize(Cost, x0, args=(nDutyCycles,timeHorizon), method="SLSQP", constraints=[C1,C_DC],bounds=bounds,jac="2-point",options={'maxiter':1000, 'disp': True,'ftol':0.05,"eps":0.03,'iprint':3,"finite_diff_rel_step":0.03})#jac=self.diff_F)  TIAGO
    res=minimize(Cost, x0, args=(nDutyCycles,timeHorizon), method='SLSQP', constraints=[C1,C_DC], bounds=bounds, jac=diff_Cost, options={'maxiter':1000, 'disp': True,'ftol':0.07,'iprint':3})
    return res
   
   
def main():
    
    nDutyCycles= 6; nInc = 1; nIncOpt = 2*nDutyCycles
    timeHorizon = 24*3600; maxInc = timeHorizon/nInc
    h0 = 4.0
    
    x = [1, 4, 7, 10., 14.5, 16.] + [2, 2.5, 2., 2., 1., 5.]
    #x2 = [1, 4, 7, 10., 14.5+0.145, 16. ]+ [2, 2.5, 2., 2., 1., 5.]
    custototal = Cost(x,nDutyCycles,timeHorizon)
    #custototal2 = Cost(x2,nDutyCycles,timeHorizon)
    #print('Custo inicial = %f' %custototal)
#
    ##Gráfico
    ##tempo_duty, level_duty, potencia_dc, pumpState_dc, cost_dc, CostT, tempo, cost_vector = converter_v2.simulation(opt_res.x,nDutyCycles,timeHorizon)
    #tempo_duty, level_duty, potencia_dc, pumpState_dc, tarif_dc, tempo, tarifa = converter_v3.simulation(x,nDutyCycles,timeHorizon)
    ##tariff_dc2 = [p*30 for p in tariff_dc]
    #tempo_duty2 = [a/3600 for a in tempo_duty] # Para ficar em Horas
    ##cost_dc2 = [p*1 for p in cost_dc]
    #custofinal = Cost(x,nDutyCycles,timeHorizon)
    #print('Custo Final', custofinal)
    #tariff = [w[0] * 30 for w in tarifa]

    #
    #t = [j/3600 for j in tempo]
    #plt.plot(tempo_duty2, level_duty, label = 'tank level')
    #plt.step(tempo_duty2, pumpState_dc, where='post', label = 'Pump status')
    ##plt.plot(tempo_duty2, cost_dc2, label = 'Cost')
    #plt.step(t, tariff, where='post')
    #plt.xlabel('Time [H]')
    #plt.xlim(xmin=0, xmax=24)
    #plt.ylabel('Tank level [m] / Tariff*30')
    #plt.grid(alpha=0.45)
    #plt.title('Custo final = %f' %custofinal)
    #plt.legend()
    #plt.show()
    
    #exit()
    
    st = time.time()
    opt_res = optimization(x0=x,nIncOpt=nIncOpt,nDutyCycles=nDutyCycles,timeHorizon=timeHorizon)
    et = time.time(); elapsed_time = et - st 
    print('Solution',opt_res.x)
    
    custofinal = Cost(opt_res.x,nDutyCycles,timeHorizon)
    print('Execution time: ', round(elapsed_time/60, 4),'minutos; Custo = ', custofinal,'€' )
    
    
    
    
    
    solution= [ 1. , 4. , 7. , 10. , 14.5, 16., 2. , 2.5, 2. , 2. , 1. , 5. ]
    custofinal = Cost(solution,nDutyCycles,timeHorizon)
    #Gráfico#    tempo_duty, level_duty, potencia_dc, pumpState_dc, tarif_dc, tempo, tarifa = converter_v3.simulation(opt_res.x,nDutyCycles,timeHorizon)
    #tempo_duty, level_duty, potencia_dc, pumpState_dc, tarif_dc, tempo, tarifa = converter_v3.simulation(solution,nDutyCycles,timeHorizon)

    #print(level_duty)
    tempo_duty2 = [a/3600 for a in tempo_duty] # Para ficar em Horas
    tariff = [w[0] * 30 for w in tarifa] # só serve para o grafico das tarifas
    imagens_tese = r'C:\Users\LENOVO\OneDrive - Universidade de Aveiro\universidade\Mestrado\2º Ano\Tese\Tese_UA\Imagens'

    t = [j/3600 for j in tempo]
    with plt.style.context('science'):
        plt.plot(tempo_duty2, level_duty, label = 'tank level')
        plt.step(tempo_duty2, pumpState_dc, where='post', label = 'Pump status')
        #plt.plot(tempo_duty2, cost_dc2, label = 'Cost')
        plt.step(t, tariff, where='post', label='Tariff')
        plt.xlabel('Time [H]')
        plt.xlim(xmin=0, xmax=24)
        plt.ylabel('Tank level [m] / Tariff*30')
        plt.grid(alpha=0.45)
        plt.title('Final cost = %f €' %custofinal)
        plt.legend()
        plt.savefig(os.path.join(imagens_tese,"optimization_fontinha.pdf"), dpi=300)
        plt.show()
        
    
   
   
if __name__ == "__main__":
    main()