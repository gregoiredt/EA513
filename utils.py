import numpy as np

def create_foyers(taille_pop, nb_foyers, borne_max, law=None):
    '''
    Crée une liste de nb_foyers qui vérifie contenant au total une population de taille_pop habitants.
    '''
    liste_foyers = [1] * nb_foyers # On initie les foyers à 1 personne
    taille_pop = taille_pop - nb_foyers # Population qu'il reste à placer

    while taille_pop > 0:
        # We choose an index according to some law
        # if law is None, the choice is uniform
        index = np.random.choice(borne_max, p=law) 
        liste_foyers[index] += 1
        taille_pop -= 1

    return liste_foyers

def concat(liste_foyers):
    m = np.max(liste_foyers)
    res = [0] * m

    for l in liste_foyers:
        res[l-1] += 1

    return res

def concat_hab(liste_foyers):
    foyers = concat(liste_foyers)
    res = [0] * len(foyers)
    for i, nb_foy in enumerate(foyers):
        res[i] =  nb_foy*(i+1)
    return res


def expand_10_law(law, factor):
    '''
    Expand a probability distribution of size 10 uniformely by multiplying it 
    by some factor.
    :args: law : Array - the probability distribution (of size 10)
    '''
    if len(law) != 10:
        raise(ValueError('Law must be of length 10'))
    
    n_law = np.zeros(10 * factor)
    for i in range(10):
        for k in range(factor):
            n_law[i*factor + k] = law[i]
    
    return n_law / np.sum(n_law)
