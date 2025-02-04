from typing import List, Tuple, Optional
import copy
import datetime
import numpy as np
import time
from functools import lru_cache
np.set_printoptions(threshold=np.inf, linewidth=400)
import math, matplotlib.pyplot as plt
from scipy.optimize import newton, minimize, Bounds, NonlinearConstraint, LinearConstraint, BFGS, differential_evolution, shgo
from scipy.integrate import solve_ivp
import joblib
import csv
from  twindow_converter import * 

def hydraulicSimulator(timeIntPumpStates,h0,iChart):
    
    # Load model ML

    model = joblib.load('xgb_model_fontinha.joblib')
    #booster = model.get_booster()# Get the Booster object from the XGBoost model
    #feature_names = booster.feature_names # Get the feature names
    #print("Input feature names:", feature_names)
    #input()
    
    # Load scalers
    scaler_in = joblib.load('scaler_in_fontinha.joblib')
    scaler_out = joblib.load('scaler_out_fontinha.joblib')
    #print(scaler_in.feature_names_in_)
    #print(scaler_out.feature_names_in_)
    
    
    
    # definição dos dicionários
    fObjRest = {'fObj': None, 'g1': [], 'g2': []}
    # Cálculo dos consumos em cada t
    # definição do polinómio para o caudal QVC
    def Q_VC(t):
        a6 = -5.72800E-05; a5 = 3.9382E-03; a4=-9.8402E-02; a3 = 1.0477; a2 = -3.8621; a1 = -1.1695; a0 = 7.53930E+01
        t = t/3600 # porque senão entra em segundos. Assim entra em horas 
        QVC = a6*(t**6)+a5*(t**5)+a4*(t**4)+ a3*(t**3)+a2*(t**2)+a1*t+a0  # original
        return QVC       
    # definição do polinómio para o caudal QR
    def Q_R(t):
        a3 = -0.004; a2 = 0.09; a1 = 0.1335; a0 = 20.0
        t = t/3600 # porque senão entra em segundos. Assim entra em horas 
        QR = a3*(t**3.)+a2*(t**2.)+a1*t + a0      
        return QR
    # definição do tarifário usando o tempo inicial do incremento/timeStep
    def tarifario(ti,tf):
        tarifHora = [None]*8; tarifCusto = [None]*8
        set(tarifHora)
        ti = ti/3600 # Para horas
        tf = tf/3600
        
        tarifHora[0]= 0; tarifCusto[0]= 0.0737
        tarifHora[1]=2; tarifCusto[1]= 0.06618 #tarifCusto[1]= 0.06618
        tarifHora[2]=6; tarifCusto[2]=  0.0737
        tarifHora[3]=7; tarifCusto[3]=  0.10094
        tarifHora[4]=9; tarifCusto[4]=  0.18581
        tarifHora[5]=12; tarifCusto[5]= 0.10094
        tarifHora[6]=24.0; tarifCusto[6]= 0.10094
        tarifHora[7]=24.9; tarifCusto[7]= 0.10094 #for the t=24.0
        tarifF = 0.
        for i in range(0, len(tarifHora)-1):
            if (ti >= tarifHora[i]) and (ti < tarifHora[i+1]):
                #Check if in middle or between tariffs
                if tf<=tarifHora[i+1]:
                    tarifF = tarifCusto[i]
                    dur = tf - ti 
                    tariffpriceInc = dur * tarifF
                elif tf >=tarifHora[i+1]:
                    dur1 = tarifHora[i+1] - ti
                    dur2 = tf - tarifHora[i+1]
                    dur_calculada = dur1 + dur2
                    print('duraçao calculada', dur_calculada)
                    #dur = tf - ti 
                    #print('duração', dur)
                    #input()
                    tariffpriceInc = (tarifCusto[i] * dur1) + (tarifCusto[i+1] * dur2)

                break
        if tarifF == 0.: print("Erro no tarifário",ti,i); quit()
        

        return tariffpriceInc
    
    # Dados gerais, constantes e caracteristicas da rede
    g = 9.81; densidade = 1000.0; densg = g * densidade
    hFixo = 100.0; hF0 = h0; #4.0; #hmin =  3; hmax = 7.0; 
    AF = 155.0; V0 = 620.0;  LPR = 3500; LRF = 6000
    f =  0.02; d =  0.3
    # variáveis constantes e definição das perdas de carga (função do caudal Q)
    #f32gpi2d5 = 32.0*f/(g*math.pi**2.0*d**5.)
    #lossesCoefPR = f32gpi2d5*LPR; lossesCoefRF = f32gpi2d5*LRF
    def hLossesPR (Q, lossesCoefPR): # Caudal em m3/s
        return lossesCoefPR * Q**2.
    def hLossesRF (Q, lossesCoefRF): # Caudal em m3/s
        return lossesCoefRF * Q**2.
    # Dados da bomba e curva hidráulica
    etaP = 0.75; pumpCoef = [280., 0, -0.0027]
    def hPumpCurve (Q,pumpCoef): # caudal em m3/h
        return pumpCoef[2]*Q**2 + pumpCoef[1]*Q + pumpCoef[0]
    # Cálculo para encontrar a raíz Qp (equilibrio da bomba vs instalação): sum(h)=0
    def BalancePump (Qp, QR, pumpCoef,lossesCoefPR,lossesCoefRF,hF,hFixo):
        return hPumpCurve(Qp,pumpCoef)- hLossesPR(Qp/3600,lossesCoefPR)-hLossesRF((Qp-QR)/3600,lossesCoefRF) - hF - hFixo        
        #
    
    # Inicialização dos vetores
    CostPrev = []; Qp3 = []
    CostPrev.append(0.0)
    #timeHorizon = 24; maxInc = 1
    #timeSteps = timeIntPumpStates['timeInt'] #Steps(x,maxInc,timeHorizon=timeHorizon)
    #pumpStateInSteps = timeIntPumpStates['pumpState'] 
    # Equações diferenciais dN/dt=(Qp-Qr.QVC)/A & dC/dt=power*tarifario
    #def difFunc(t, y, pumpState , AF,pumpCoef,lossesCoefPR,lossesCoefRF,hFixo, densg, etaP):
    
    
    tank_final=[]
    potencia = []
    empty_timeIncrem = {
            "number":None,
            "startTime":None,
            "duration":None,
            "endTime":None,
            #"hFini":None,
            #"hFfin":None
            };
    
    timeInc=[]; pump_state = []; demands1 = []; demands2 = [];
    columns_to_normalize1 = ['tank_0', 'junction_1', 'junction_2']
    columns_to_normalize2 = ['Δtank_0', 'pumps_kw']
    
    for i in range(len(timeIntPumpStates['timeInt']) - 1):
        timeInc.append(empty_timeIncrem.copy())
        #timeInc[i]['number'] = i + 1
        
        QR = Q_R(timeIntPumpStates['timeInt'][i])
        QVC = Q_VC(timeIntPumpStates['timeInt'][i])
        nivel_inicial = h0
        
        junction_1 = QVC
        junction_2 = QR
        pumps_0 = timeIntPumpStates['pumpState'][i]
        
        if timeIntPumpStates['timeInt'][i] == 0:
            final = nivel_inicial # Se for o primeiro ponto
            power = 0
            dtime = (timeIntPumpStates['timeInt'][i + 1] - timeIntPumpStates['timeInt'][i])
        else: 
            tank_0 = tank_final[-1]
            x_test = pd.DataFrame([[tank_0, pumps_0, junction_1, junction_2]], columns=['tank_0', 'pumps_0', 'junction_1', 'junction_2'])        
            dtime = (timeIntPumpStates['timeInt'][i + 1] - timeIntPumpStates['timeInt'][i]) # !! Atençao segundos ou horas
            # Normalização
            x_normalizado = x_test.copy()
            x_normalizado[columns_to_normalize1] = scaler_in.transform(x_normalizado[columns_to_normalize1])       
            # prediction
            y_pred = model.predict(x_normalizado)
            y_pred_df = pd.DataFrame(data=y_pred, columns=[columns_to_normalize2])
            #desnormalizar
            y_pred_un = scaler_out.inverse_transform(y_pred_df[columns_to_normalize2])
            # Integration
            final = y_pred_un[0,0] * dtime + tank_0 # Final tank level
            power = y_pred_un[0,1]  # Potencia da bomba 
        
            
            
        timeInc[i]['startTime'] = timeIntPumpStates['timeInt'][i]
        timeInc[i]['endTime'] = timeIntPumpStates['timeInt'][i + 1]
        timeInc[i]['duration'] = dtime
        #timeInc[i]['hFini'] = tank_0
        #timeInc[i]['hFfin'] = tank_final[i]
        timeInc[i]['TariffInc'] = tarifario(timeInc[i]['startTime'], timeInc[i]['endTime'] )
        tank_final.append(final)
        potencia.append(power)
        pump_state.append(pumps_0)
        
        # extra
        demands1.append(junction_1)
        demands2.append(junction_2)
        

        
    return tank_final, potencia, timeInc, pump_state, demands1, demands2



def simulation(x,nDutyCycles,timeHorizon):

   
    #my_converter.visualizer_disc(control_array=pump_status[0], control_time=time_array)
    maxInc = 3600/12 # 5min 
    x_sec = [j*3600 for j in x ]

    # Converter 
    
        # definição dos dicionários: times2integrate & pumpStates
    timeIntPumpStates = {'timeInt': [], 'pumpState': []}

    # criação da lista de timeSteps pelo x e maxInc, coloca por ordem e retira duplicados
    def Steps(x_sec,nDutyCycles,maxInc,timeHorizon):
        # tomando o x para todas as horas 
        steps1=[]
        for i in range (0,nDutyCycles): 
            steps1.append(x_sec[i]) # start of dutycycle 
            steps1.append(x_sec[i]+x_sec[nDutyCycles+i]) # end of dutycycle        
        return sorted(set(([round(p*maxInc,10) for p in range(0, int(timeHorizon/maxInc))] + [24*3600] + steps1)))
 
    def pumpStateM3 (x,nDutyCycles,t): #função "x to state converter" para len(x)=24
        s = 0
        for i in range(0, nDutyCycles):
            if (t >= x_sec[i] and t < x_sec[i]+x_sec[nDutyCycles+i]): s = 1 #print(t,x[i],x[i]+x[nDutyCycles+i],s)
        return s

    def pumpstatesList(x_sec,nDutyCycles,timeSteps):
        pumpStateInSteps=[]
        for valueList in timeSteps:
            pumpStateInSteps.append(pumpStateM3(x_sec,nDutyCycles,valueList))
        return pumpStateInSteps

    timeIntPumpStates['timeInt'] = Steps(x_sec,nDutyCycles,maxInc,timeHorizon=timeHorizon)
    timeIntPumpStates['pumpState'] = pumpstatesList(x_sec,nDutyCycles,timeSteps=timeIntPumpStates['timeInt'])
    

    def verificar_nivel_dutycycle(x_sec, tempos, niveis, potencia, estados_bomba, cost_vector):
        respostas = []

        # Iterar sobre os tempos
        for t in tempos:
            nivel = None

            # Verificar inicio do dia
            if t == 0 :
                indice = tempos.index(t)
                nivel = niveis[indice]
                power = potencia[indice]
                state = estados_bomba[indice]
                Cost = cost_vector[indice]
                
                
            # Verificar início do dutycycle
            if t in x_sec[:len(x_sec)//2]:
                indice = tempos.index(t)
                nivel = niveis[indice]
                power = potencia[indice]
                state = estados_bomba[indice]
                Cost = sum(cost_vector[indice-1:indice])

            # Verificar fim do dutycycle
            for  i, (inicio, duracao) in enumerate(zip(x_sec[:len(x_sec)//2], x_sec[len(x_sec)//2:])):
                fim = inicio + duracao
                if duracao == 0 or duracao <= 0.0001:
                    if t == inicio:  # dutycycle com duração zero
                        indice = tempos.index(t)
                        nivel = niveis[indice]
                        power = potencia[indice]
                        state = estados_bomba[indice]
                        Cost = sum(cost_vector[indice-1:indice])
                        respostas.append((t, nivel, power, state, Cost))
                    
                else:
                    if t == fim:  # fim dutycycle
                        indice = tempos.index(t)
                        nivel = niveis[indice]
                        power = potencia[indice]
                        state = estados_bomba[indice]
                        Cost = sum(cost_vector[indice-1:indice])
                    
            if nivel is not None:
                respostas.append((t, nivel, power, state, Cost))

        # Adicionar último nível após o último tempo
        ultimo_nivel = niveis[-1]
        ultimo_power = potencia[-1]
        ultimo_state = estados_bomba[-1]
        ultimo_cost = cost_vector[-1]
        respostas.append((86400, ultimo_nivel, ultimo_power, ultimo_state, ultimo_cost))
        
        tempos_dc = []
        niveis_dc = []
        potencia_dc = []
        p_state_dc = []
        cost_dc = []

        for tupla in respostas:
            tempos_dc.append(tupla[0])
            niveis_dc.append(tupla[1])
            potencia_dc.append(tupla[2])
            p_state_dc.append(tupla[3])
            cost_dc.append(tupla[4])
        
        return tempos_dc, niveis_dc, potencia_dc, p_state_dc, cost_dc

      

    #Simulation
    
    
    # Implementação modelo no benchmark 
    h0 = 4.0 #Initial tank level 

    tank_final, potencia, timeInc, pump_state, demands1, demands2 = hydraulicSimulator(timeIntPumpStates, h0=h0, iChart=1)
    tempo = [entry['startTime'] for entry in timeInc] # todos os tempos da key 'startTime'
    tariffpriceInc = [entry['TariffInc'] for entry in timeInc] # todas as tarifas da key 'Tariff'
    
    
    
    def Custo(tempo, potencia, tariffpriceInc):
        CostT = 0; cost_vector = [];
        #energy = [round(c,1) for c in potencia]
        energy = potencia
        for i in range(len(tempo)):
            cost = energy[i] * tariffpriceInc[i] 
            CostT += cost
            cost_vector.append(cost)
        return CostT, cost_vector

    CostT, cost_vector = Custo(tempo, potencia, tariffpriceInc)

    # Voltar para formato dutycycle
    tempos_dc, niveis_dc, potencia_dc, p_state_dc, cost_dc = verificar_nivel_dutycycle(x_sec, tempos=tempo, niveis=tank_final, potencia=potencia, estados_bomba=pump_state,cost_vector=cost_vector )
    return tempos_dc, niveis_dc, potencia_dc, p_state_dc, cost_dc, CostT, tempo, cost_vector




#x = [1, 4, 7, 10., 14.5, 16.] + [2, 2.5, 2., 2., 1., 5.]
#nDutyCycles= 6; timeHorizon = 24*3600;
#
#
#tank_final, potencia, timeInc, pump_state = simulation(x,nDutyCycles,timeHorizon)


#
#print('tariffs  in dutycycles', tariff_dc)
#input()
#
#
#print('tank level  in dutycycles', level_duty)
#print('Pump status in dutycycles', pumpState_dc)
#print('Power in dutycycles', potencia_dc)
#print('time of dutycycles' , [t/3600 for t in tempo_duty] )
#input()
#
#
#plt.plot(tempo_duty, level_duty, label = 'tank level')
#plt.step(tempo_duty, pumpState_dc, where='post')
#plt.legend()
#plt.show()