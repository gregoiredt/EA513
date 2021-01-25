import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.stats import linregress
from scipy.optimize import fsolve
from scipy.special import binom
import matplotlib.pyplot as plt

#Debug
import pdb

# Utils to Execute on IPython
from IPython.display import display, clear_output

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

CHANGE_REC = [0, -1, 1]
CHANGE_INF = [-1, 1, 0]
TYPES = ['S', 'I', 'R']

class Simulation():
    '''
    Simulation du modèle basé sur les EDO
    variables :
    * T : end time of the simulation
    * thresh : if in kwargs, the simulation ends after the number of infected is greater than thresh
    '''
    
    def __init__(self, beta_f, beta_m, gamma, liste_foyer, T, n, dic_infection, id_sim=0, **kwargs):
        # Allow either for a threshold by number of infected or by time
        self.time_stop = True
        if 'thresh' in kwargs:
            self.thresh = kwargs['thresh']
            self.time_stop = False
        # Initialisation des autres paramètres
        self.beta_f = beta_f
        self.gamma = gamma
        self.beta_m = beta_m
        self.liste_foyer = liste_foyer
        self.T = T
        self.n = n
        self.dt = T/n
        self.id_sim = id_sim
        self.dic_infection = dic_infection
        self.M = len(liste_foyer) # Nombre maximal de résident dans un foyer
        self.state_0 = kwargs['state_0'] if 'state_0' in kwargs else None  
   
    def get_df(self):
        return self.df
    
    def next_state(self, xs, xi, xr, dict_types, return_foyers = False):
        next_xs = np.zeros(len(dict_types)+1)
        next_xi = np.zeros(len(dict_types)+1)
        next_xr = np.zeros(len(dict_types)+1)
        
        if return_foyers:

            foyers_s = np.zeros(self.M+1)
            foyers_i = np.zeros(self.M+1)
            foyers_r = np.zeros(self.M+1)

        for ms in range(self.M+1):
            for mi in range(self.M+1-ms):
                for mr in range(self.M+1-mi-ms):
                    indice = dict_types[(ms,mi,mr)]
                    xis = 0
                    xri = 0
                    xss = 0
                    if ms<self.M and mi >0:
                        xis = xi[dict_types[(ms+1,mi-1,mr)]]
                        xss = xs[dict_types[(ms+1,mi-1,mr)]]

                    if mi<self.M and mr >0:
                        xri = xi[dict_types[(ms,mi+1,mr-1)]]

                    xs_prime = self.beta_f*ms*((ms+1)*xis - ms*xi[indice]) + self.gamma*ms*(xri - xi[indice])+ ms*self.beta_m*(xss - xs[indice])*np.sum(xi)/(np.sum(xr)+np.sum(xs)+np.sum(xi))
                    xi_prime = self.beta_f*mi*((ms+1)*xis - ms*xi[indice]) + self.gamma*mi*(xri - xi[indice])+ mi*self.beta_m*(xss - xs[indice])*np.sum(xi)/(np.sum(xr)+np.sum(xs)+np.sum(xi))
                    xr_prime = self.beta_f*mr*((ms+1)*xis - ms*xi[indice]) + self.gamma*mr*(xri - xi[indice]) #+ mr*self.beta_m*(xss - xs[indice])*np.sum(xi)/(np.sum(xr)+np.sum(xs)+np.sum(xi))


                    next_xs[indice] = xs[indice] + self.dt*xs_prime
                    next_xi[indice] = xi[indice] + self.dt*xi_prime
                    next_xr[indice] = xr[indice] + self.dt*xr_prime

                    if return_foyers:
                        foyers_s[ms+mi+mr] += next_xs[indice]
                        foyers_i[ms+mi+mr] += next_xi[indice]
                        foyers_r[ms+mi+mr] += next_xr[indice]

        return (next_xs,next_xi,next_xr)
    
    def initialize_and_infect(self, dict_types):
        '''
        Crée le dictionnaire d'origine et l'infect à partir du dictionnaire dic_infection donné.
        Dic_types correspond à l'ensemble des types possibles.
        '''

        # Initialisation

        M = self.M
        XS = np.zeros(len(dict_types)+1)
        XI = np.zeros(len(dict_types)+1)
        XR = np.zeros(len(dict_types)+1)
        
        LIST_XS = [XS]
        LIST_XI = [XI]
        LIST_XR = [XR]

        # Prise en compte de la population des foyers

        for ms in range(1,M+1):
            XS[dict_types[(ms,0,0)]] = self.liste_foyer[ms-1]
        
        # Infection (on rajoute)
        
        for key in self.dic_infection.keys():
            number = self.dic_infection[key]
            XI[dict_types[key]] = number
            XS[dict_types[key]] = number * (key[0]//key[1])

        return XS, XI, XR, LIST_XS, LIST_XI, LIST_XR
    
    def initalize_from_state(self, dict_types):
        '''
        Initialise non pas à partir d'un dictionnaire d'infection, mais
        à partir d'un état selon la forme des états fournis par
        la classe StochasticSimulation
        '''
        if not(hasattr(self, 'state_0')):
            raise ValueError('No state has been given. Please ensure that the attribute state_0 is filled')
        
        df = self.state_0
        # Initialisation

        XS = np.zeros(len(dict_types)+1)
        XI = np.zeros(len(dict_types)+1)
        XR = np.zeros(len(dict_types)+1)
        
        # On modifie chaque état en fonction du state_0 donneé
        for key in dict_types.keys():

            ms, mi, mr = key
            row = df[(df.ms == ms) & (df.mi == mi) & (df.mr == mr)]

            row_S, row_I, row_R = row[row.etat == 'S'], row[row.etat == 'I'], row[row.etat == 'R']
            key = (ms, mi, mr)
            if len(row_S) > 0:
                XS[dict_types[key]] = row_S.nombre.iloc[0]
            if len(row_I) > 0:
                XI[dict_types[key]] = row_I.nombre.iloc[0]
            if len(row_R) > 0:
                XR[dict_types[key]] = row_R.nombre.iloc[0]

        LIST_XS = [XS]
        LIST_XI = [XI]
        LIST_XR = [XR]

        return XS, XI, XR, LIST_XS, LIST_XI, LIST_XR

    def simulate(self, from_given_state=False):
        '''
        Simule toute la propagation de l'épidémie
        '''

        dict_types = self.initialize_dict()

        if from_given_state:
            # Initialise à partir d'un état donné selon un format précis
            XS, XI, XR, LIST_XS, LIST_XI, LIST_XR = self.initalize_from_state(dict_types)

        else:
            # Initialise avec le dictionnaire des infectés
            XS, XI, XR, LIST_XS, LIST_XI, LIST_XR = self.initialize_and_infect(dict_types)

        if self.time_stop:
            # Simulation en temps habituelle
            for k in range(self.n):
                XS,XI,XR = self.next_state(LIST_XS[-1],LIST_XI[-1],LIST_XR[-1], dict_types)
                LIST_XS.append(XS)
                LIST_XI.append(XI)
                LIST_XR.append(XR)
                # Affichage
                if k % 10 == 0:
                    clear_output(wait=True)
                    display(f'EDO Simulation in progress : {100 * k / self.n} % completed')

        else:
            # Simulation avec arrêt si dépassement de la borne max
            for k in range(self.n):
                XS,XI,XR = self.next_state(LIST_XS[-1],LIST_XI[-1],LIST_XR[-1], dict_types)
                LIST_XS.append(XS)
                LIST_XI.append(XI)
                LIST_XR.append(XR)
                
                # Affichage :
                if k % 10 == 0:
                    clear_output(wait=True)
                    display(f'EDO Simulation in progress : {100 * k / self.n} % completed')

                if np.sum(XI) > self.thresh:
                    # On dépasse la limite
                    self.time_thresh = k * self.T / self.n
                    # Changement de variable qui évite les erreurs dans les calculs des infos utiles.
                    self.T = self.time_thresh
                    clear_output(wait=True)
                    display(f'EDO Simulation finished : {np.sum(XI)} people have been infected at time {self.time_thresh}')
                    break


        NB_S = [np.sum(xs) for xs in LIST_XS]
        NB_I = [np.sum(xi) for xi in LIST_XI]
        NB_R = [np.sum(xr) for xr in LIST_XR]
        
        self.NB_S = NB_S
        self.NB_I = NB_I
        self.NB_R = NB_R

        df = pd.DataFrame({'susceptibles' : NB_S,
                            'infectés': NB_I,
                            'remis': NB_R,
                            'foyer': [self.id_sim]*len(NB_S)},
                            index = [k for k in range(len(NB_S))])
        df = df.melt(id_vars = ['foyer'], value_vars=['susceptibles', 'infectés', 'remis'])
        df.reset_index(inplace=True)
        df.columns = ['step', 'foyer', 'variable', 'value']
        df.step = df.step % (self.n + 1)
        df['temps'] = df['step']* self.dt
        
        self.df = df

        return df 
    
    def show_interest_variables(self):

        if not(hasattr(self, 'df')):
            self.df = self.simulate()
        df = self.df

        # Initialisation des paramètres
        index_foyer = list(df.foyer.unique())
        max_index = df.step.max()

        # Calcul des variables d'intérêt
        list_greatest_slopes = [self.get_greatest_slope(foyer=i, max_index=max_index) for i in index_foyer]
        list_infectes_max = [self.get_infecte_max(foyer=i) for i in index_foyer]
        list_R0 = [self.get_r0(foyer=i) for i in index_foyer]
        nb_foyer = len(index_foyer)

        df_res = pd.DataFrame({
          'foyer' : [i for i in index_foyer],
          'nb_infectes_max' : [list_infectes_max[i][0] for i in range(nb_foyer)],
          'temps_nb_infecte_max' : [list_infectes_max[i][1] for i in range(nb_foyer)],
          'pente_infectes_max' : [list_greatest_slopes[i][0] for i in range(nb_foyer)],
          'temps_pente_max' : [list_greatest_slopes[i][1] for i in range(nb_foyer)],
          'nb_total_infectes' : [self.get_total_infected_people(foyer=i) for i in index_foyer],
          'R_0': [list_R0[i] for i in range(nb_foyer)]
          }, index = index_foyer)
        
        # Saving in the simulation variable
        self.interest_variables = df_res

        return df_res
    
    def get_all_infos(self):

        if not(hasattr(self, 'df')):
            self.df = self.simulate()
        
        df_complete = pd.DataFrame({
            'id' : self.id_sim,
            'beta_f' : self.beta_f,
            'gamma' : self.gamma,
            'beta_m' : self.beta_m,
            'foyer' : str(self.liste_foyer),
            'CI' : str(self.dic_infection)}, index = [self.id_sim])
        
        df_VI = self.show_interest_variables()
        
        self.all_infos = pd.concat([df_complete, df_VI], axis=1)
        
        return self.all_infos

    def get_infecte_max(self, foyer):
        df= self.df
        res = df[(df.variable == 'infectés') & (df.foyer == foyer)]
        return res.value.max(), res[res.value == res.value.max()].temps.iloc[0]

    def get_total_infected_people(self, foyer=1):
        df = self.df
        return df[(df.temps == df.temps.max()) & (df.foyer == foyer) & (df.variable.isin(['infectés', 'remis']))].value.sum()

    def get_greatest_slope(self, foyer=1, max_index=1000):
        df = self.df
        df = df[(df.foyer == foyer) & (df.variable == 'infectés')]
        slopes = [0]
        for i in range(1, max_index):
            slopes.append(df.loc[df.step == i, 'value'].iloc[0] - df.loc[df.step == i-1, 'value'].iloc[0])
            
        self.step_greatest_slope = np.argmax(slopes)
        self.time_greatest_slope = df[df.step == np.argmax(slopes)].temps.iloc[0]
        return np.max(slopes), self.time_greatest_slope
    
    def get_r0(self, foyer=1):
        df = self.df
        t_max = self.time_greatest_slope * self.dt
                       
        if t_max < 5 * self.dt:
            # Il n'y a presque pas d'explosion, l'épidémie s'éteint dès le début...
            return np.nan
        
        # On ne veut que le comportement exponentiel, on ne garde donc que la partie avant le nombre maximal d'infectés.
        data = df[(2 * self.dt <df['temps']) & (df['temps']< 0.9 * t_max) & (df['variable']=='infectés')]
        data.loc[:,('log_value')] = np.log(data['value'])
        R0 = linregress(data[data.foyer == foyer]['temps'], data[data.foyer == foyer]['log_value']).slope+1
        return R0

    def plot(self, xlim):
        plt.plot(self.NB_I, label="infectés", color ='r')
        plt.plot(self.NB_S, label="susceptible", color='y')
        plt.plot(self.NB_R, label="remis", color='g')
        plt.xlim([0,xlim])
        plt.show()

    def initialize_dict(self):
        M = self.M
        dict_types = {}
        indice = 0
        for ms in range(M+1):
            for mi in range(M+1-ms):
                for mr in range(M+1-mi-ms):
                    dict_types[(ms,mi,mr)] = indice
                    indice+=1
        return dict_types

    def get_temps_atteinte(self, valeur):
        df2 = self.df[self.df.variable=='infectés'].copy()
        try:
            return df2[df2.value>=10].head(1).temps.iloc[0]
        except IndexError as _:
            raise(IndexError("Le niveau inséré n'est pas atteint par la simulation"))

        
class StochasticSimulation(Simulation) :
    '''
    Cette classe hérite de la classe Simulation défiie au dessus et
    implémente la modélisation stochastique en plus
    '''
    
    def __init__(self, beta_f, beta_m, gamma, liste_foyer, T, n, dic_infection, id_sim=0, **kwargs):
        super().__init__(beta_f, beta_m, gamma, liste_foyer, T, n, dic_infection, id_sim, **kwargs)
        self.tab_time = [0]
        # Variable that will check if the number of infected goes above the thresh
        self.below_thresh = True
        # Variable used to let the code be faster 
        self._nb_infected = 0

    def recovery(self, state, m):
        '''
        Modify the state following a recovery in an household of type m.
        Parameters:
        State (dict) -- current state of the system
        m (3uple) -- type of household infected 
        '''
        n_state = state.copy()

        # On retire tout le foyer de chaque aggrégat
        for i in range(len(TYPES)):
            n_state = self.change_state(n_state, f'{TYPES[i]}_{m[0]}_{m[1]}_{m[2]}', -m[i])

        # On ajoute le nouveau foyer aux aggrégats correspondant
        change = [0, -1, 1]
        for i in range(len(TYPES)):
            n_state = self.change_state(n_state, f'{TYPES[i]}_{m[0]}_{m[1]-1}_{m[2]+1}', m[i] + change[i])

        self._nb_infected -= 1
        return n_state

    def infection(self, state, m):
        '''
        Modify the state following an infection in an household of type m.
        Parameters:
            State (dict)-- current state of the system
            m (3uple)-- type of households infected 
        '''
        n_state = state.copy()
        # On retire tout le foyer de chaque aggrégat
        for i in range(len(TYPES)):
            n_state = self.change_state(n_state, f'{TYPES[i]}_{m[0]}_{m[1]}_{m[2]}', -m[i]) 

        # On ajoute le nouveau foyer aux aggrégats correspondant
        change = [-1, 1, 0]
        for i in range(len(TYPES)):
            n_state = self.change_state(n_state, f'{TYPES[i]}_{m[0]-1}_{m[1]+1}_{m[2]}',  m[i] + change[i])
        
        self._nb_infected +=1
        return n_state

    def one_turn(self, df_state, counter):
        '''
        One step in the system, this function also takes care of the output
        '''

        try:
            df_state = self.next_state(df_state)
        except TypeError:
            # Il n'y a plus d'infectés, on fixe donc le nombre d'infectés et on retourne le df inchangé
            self._nb_infected=0
            self.l_state.append(df_state)
            return df_state
        else:
            self.l_state.append(df_state)
            # Affichage :
            if counter % 10 == 0:
                clear_output(wait=True)
                display(f'Stochastic Simualation in progress : {100 * self.tab_time[-1] / self.T} % completed')
            return df_state
    
    def simulate(self):
        df_state =  self.initialize_and_infect()
        self.l_state = [df_state]
        counter = 0
        
        if not(self.time_stop):
            # Cas avec limite 
            while self.tab_time[-1] < self.T and self._nb_infected>0:
                df_state = self.one_turn(df_state, counter)
                counter += 1
                if self._nb_infected >= self.thresh:
                    #self.thresh is not called if time stop is True
                    self.time_thresh = self.tab_time[-1]
                    # Again, T is being changed to facilitate other calculations.
                    self.T = self.time_thresh
                    clear_output(wait=True)
                    display(f'Stochastic Simualation finished : {self.thresh} people have been infected at time {self.time_thresh}.')
                    break
                counter +=1
        else:
            #Cas sans limite
            while self.tab_time[-1] < self.T and self._nb_infected>0:
                df_state = self.one_turn(df_state, counter)
                counter += 1
        
        NB_I = [self.get_nb_etat(state, 'I') for state in self.l_state]
        NB_R = [self.get_nb_etat(state, 'R') for state in self.l_state]
        NB_S = [self.get_nb_etat(state, 'S') for state in self.l_state]

        # Enregistrement de ces variables dans la classe :

        self.NB_S = NB_S
        self.NB_I = NB_I
        self.NB_R = NB_R
        
        if len(NB_I)>len(self.tab_time):
            # La simualtion n'a plus d'infectés, on ne prend pas le dernier élément qui s'est répété
            self.NB_S=self.NB_S[:-1]
            self.NB_I=self.NB_I[:-1]
            self.NB_R=self.NB_R[:-1]

        df = pd.DataFrame({'susceptibles' : NB_S,
                            'infectés': NB_I,
                            'remis': NB_R,
                            'foyer': [self.id_sim]*len(NB_S),
                            'temps': self.tab_time},
                            index = [k for k in range(len(NB_S))])

        df = df.melt(id_vars = ['foyer', 'temps'], value_vars=['susceptibles', 'infectés', 'remis'])
        df.reset_index(inplace=True)
        df.columns = ['step', 'foyer', 'temps', 'variable', 'value']
        df.step = df.step % (len(NB_S))  
        self.df = df

        return df 

    def draw_time(self, state):
        '''
        Draw all exponetial times for the system, compares them and gives
        the lowest (non zero) one
        '''
        nombre_s = state.loc[state.etat == 'S', 'nombre'].sum()
        nombre_i = state.loc[state.etat == 'I', 'nombre'].sum()
        if nombre_i == 0: 
            # Il n'y a plus d'infectés, on dit qu'il faut s'arrêter
            return None
        
        taux_champs_moyen = self.beta_m * nombre_s * nombre_i / np.sum(self.liste_foyer)
 
        somme = taux_champs_moyen # Sert à calculer la somme de tous les taux

        law_on_m = pd.DataFrame({'transition_type': 'champs_moyen',
                        'ms': 0,
                        'mi': 0,
                        'mr': 0,
                        'rate': taux_champs_moyen},
                        columns=['transition_type', 'ms', 'mi', 'mr', 'rate'],
                        index=[0])

        for _, row in state[(state.etat == 'I') & state.nombre>0].iterrows():
            ms, mi, mr, pop_foyer = row.ms, row.mi, row.mr, row.nombre

            lambda_1 = (self.beta_f * pop_foyer * ms) # Infectionx  
            lambda_2 = (self.gamma * pop_foyer) # Recovery

            new_line_1 = pd.DataFrame({'transition_type': 'infection',
                        'ms': ms,
                        'mi': mi,
                        'mr': mr,
                        'rate': lambda_1},
                        index=[0], columns=['transition_type', 'ms', 'mi', 'mr', 'rate'])
            
            new_line_2 = pd.DataFrame({'transition_type': 'recovery',
                        'ms': ms,
                        'mi': mi,
                        'mr': mr,
                        'rate': lambda_2},
                        index=[0], columns=['transition_type', 'ms', 'mi', 'mr', 'rate'])

            somme += lambda_1 + lambda_2
            law_on_m = pd.concat([law_on_m, new_line_1, new_line_2], ignore_index=True)

        time = np.random.exponential(scale= 1 / somme) # Temps exponentiel choisi

        # On cherche à quel état l'appliquer
        # On définit une bonne proba sur l'ensemble
        law_on_m['rate'] = law_on_m['rate'] / somme
        law_on_m.rate = law_on_m.rate.astype(float)

        # Choix de l'état selon la bonne proba
        index_chosen = np.random.choice(a=np.array(law_on_m.index), size=1, p=np.array(law_on_m.rate))[0]
        if index_chosen == 0:
            # On choisit un foyer avec une probilité uniforme en fonction des populations des foyers
            state_with_proba = state[state.etat == 'S'].copy()
            state_with_proba['prob'] = state_with_proba['nombre'].astype(float) / nombre_s
            state_with_proba.reset_index(inplace=True)
            index_chosen = np.random.choice(a=np.array(state_with_proba.index),
                                            size=1,
                                            p=np.array(state_with_proba.prob))[0]
            foyer_chosen = state_with_proba.iloc[index_chosen]
            m_chosen = [foyer_chosen.ms, foyer_chosen.mi, foyer_chosen.mr, 'infection']

        else:
            foyer_chosen = law_on_m.iloc[index_chosen]
            m_chosen = [foyer_chosen.ms,
                        foyer_chosen.mi,
                        foyer_chosen.mr,
                        foyer_chosen.transition_type]

        return (time, m_chosen)
       
    def change_state(self, state, key, added_value):
        '''
        Check that current dict is not empty, may be useful for optimization
        '''
        try:
            state.loc[key, 'nombre'] += added_value
        except (KeyError) as _ :
            key_s = key.split('_')

            new_line = pd.DataFrame({
            'etat': key_s[0],
            'ms': int(key_s[1]),
            'mi': int(key_s[2]),
            'mr': int(key_s[3]),
            'nombre': added_value
            }, columns=['etat', 'ms', 'mi', 'mr', 'nombre'], index=[key])

            state = pd.concat([state, new_line])
            # Faire attention, peut être négative
            
        # On supprime l'entrée parce qu'elle est vide
        index_to_drop = set(state[state.nombre<0].index)
        m_below_zero = {l[2:] for l in index_to_drop}
        # On drop tous les types du foyer vide
        m_to_drop = {f'{_type}_{m}' for m in m_below_zero for _type in ['S', 'I', 'R']} 
        state = state.drop(index=m_to_drop, errors='ignore')

        return state

    def initialize_and_infect(self):
        '''
        Calcul l'état intitial et infecte suivant l'infection donnée.
        '''
        state_0 = pd.DataFrame(columns=['etat', 'ms', 'mi', 'mr', 'nombre'])
        # taille du foyer
        for k in range(len(self.liste_foyer)):
            new_line = pd.DataFrame({
                'etat': 'S',
                'ms': k+1,
                'mi': 0,
                'mr': 0,
                'nombre': self.liste_foyer[k]
                }, columns=['etat', 'ms', 'mi', 'mr', 'nombre'], index=[f'S_{k+1}_0_0'])
            state_0 = pd.concat([state_0, new_line])

        # Modification of the state according to the infections given
        for m in self.dic_infection.keys():
            m_before = (m[0]+1, m[1]-1, m[2]) # Dans le dictionnaire m est le résultat voulu, ici on veut le foyer à infecter
            nb_infection = self.dic_infection[m]
            for _ in range(nb_infection):
                state_0 = self.infection(state_0, m_before)
        
        return state_0

    def next_state(self, state):
        '''
        Simulate an evolution of one step of the whole system
        '''
        time = self.tab_time[-1]


        try:
            n_time, m = self.draw_time(state)
        except TypeError:
            # Cela veut dire qu'il n'y a plus d'infectés:
            raise TypeError("Il n'y a plus d'infectés")
        else:
            change_type = m[-1] # Type de changement, on pourra simplifier ça plus tard
            m = m[0:3].copy()
            if change_type == 'recovery':
                n_state = self.recovery(state, m)
            if change_type == 'infection':
                n_state = self.infection(state, m)
            
            # MaJ de la table temporelle
            self.tab_time.append(time + n_time)

            return n_state
    
    def to_dic_infected(self, tau) :
        '''
        Renvoie le dictionnaire des infectés sous la borne forme,
        correspondant au temps tau donné (prend le premier supérieur à tau)
        '''

        # On récupère le temps voulu
        df = self.df[['step', 'temps']]
        n_df = df[df.temps >= tau].sort_values(by='temps')
        
        try:
            step = n_df.step.iloc[0]
        except (IndexError):
            raise IndexError(f'Tau is greater than the last time of the Simulation ({df.temps.max()})')
            
        # On récupère le dataframe du temps voulu
        df_state = self.l_state[step]     
        df_state_infected = df_state[df_state.etat == 'S']
        dic_infected = {}
        for _, row in df_state_infected.iterrows():
            # Attention : ici nombre ne représente que le nombre d'infecté... 
            ms, mi, mr, pop_foyer = row.ms, row.mi, row.mr, row.nombre
            dic_infected[(ms, mi, mr)] = pop_foyer / mi
        
        return dic_infected

    def give_last_state(self):
        return self.l_state[-1]

    def get_nb_etat(self, state, etat):
        '''
        Renvoie le nombre de personne d'un état etat à partir du
        dataframe state.
        '''
        return state.loc[(state.etat == etat), 'nombre'].sum()


class DualSimulation(Simulation):
    '''
    Associe les deux types de simulation (EDO et stochastique)
    Two types of threshold can be added : 
        * 'thresh_sto' : the stochastic simulation ends when this threshold is exceeded.
        * 'thresh' : the whole simulation stops if the level is exceeded.
    '''

    def __init__(self, beta_f, beta_m, gamma, liste_foyer, T_sto, T, n, dic_infection, id_sim=0, **kwargs):
        super().__init__(beta_f, beta_m, gamma, liste_foyer, T, n, dic_infection, id_sim, **kwargs)
        self.T_sto = T_sto
        # Simulation de la partie stochastique
        thresh_sto_sim = True # Booleen permettant de savoir quelle type de limite on impose
        if 'thresh_sto' in kwargs:
            # Simulation avec arrêt par limite
            self.thresh_sto = kwargs['thresh_sto']
            self._stochastic_sim = StochasticSimulation(beta_f, beta_m, gamma, liste_foyer, T_sto, n, dic_infection, id_sim, thresh=self.thresh_sto)
        else:
            # Simulation classique
            self._stochastic_sim = StochasticSimulation(beta_f, beta_m, gamma, liste_foyer, T_sto, n, dic_infection, id_sim, **kwargs)
            thresh_sto_sim = False
        
        self._stochastic_sim.simulate()
        self.tab_time_sto = self._stochastic_sim.tab_time
        self.df_sto = self._stochastic_sim.get_df()
        if not(thresh_sto_sim) and self.df_sto.temps.max() < T_sto :
            # On a dépassé le threshold dans le cas global
            self.df = self.df_sto
        else:
            self.state_0 = self._stochastic_sim.give_last_state()
            last_temps, last_step = self.df_sto.temps.max(), self.df_sto.step.max() # On rapelle que T_sto < last_temps

            # Simulation de la partie EDO
            _edo_sim = Simulation(beta_f, beta_m, gamma, liste_foyer, T - last_temps, n, dic_infection, id_sim, state_0=self.state_0, **kwargs)
            _edo_sim.simulate(from_given_state=True)
            self._edo_sim = _edo_sim
            df_edo = _edo_sim.get_df()
            df_edo.step = df_edo.step + last_step
            df_edo = df_edo[df_edo.step > last_step] # On évite la redondance de l'état de jointure
            df_edo.temps = df_edo.temps + last_temps
            self.df_edo = df_edo
            self.tab_time_edo = [ T_sto + i * (T - T_sto) / n for i in range(n)]

        # Mise en commun
        self.df = pd.concat([self.df_sto, self.df_edo], axis=0)
    
    def simulate(self):
        raise SystemError('You should not call such a function on this particular class')

    def plot(self, **kwargs):
        '''
        Plot the graph of the simulation
        '''
        xlim = kwargs['xlim'] if 'xlim' in kwargs else self.T
        if not(hasattr(self, 'df')):
            raise SystemError('You must simulate the model first !')
        sns.lineplot(data=self.df, x='temps', y='value', hue='variable')
        plt.xlim((0, xlim))
        plt.show()

         

    


