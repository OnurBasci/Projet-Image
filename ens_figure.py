def est_a_linterieur(point, polygone):
    """
    Détermine si un point est à l'intérieur d'un polygone en utilisant la méthode de ray casting.
    """
    nb_intersections = 0

    for i in range(len(polygone)):
        p1 = polygone[i]
        p2 = polygone[(i + 1) % len(polygone)]

        if ((p1[1] > point[1]) != (p2[1] > point[1])) and (point[0] < (p2[0] - p1[0]) * (point[1] - p1[1]) / (p2[1] - p1[1]) + p1[0]):
            nb_intersections += 1
    
    return nb_intersections % 2 == 1

def point_dans_figures(point, figures):
    """
    Détermine si un point est a l'interieur des figures
    """
    # Pour chaque figure
    for figure in figures:
        # Si le point est à l'intérieur de la figure
        if est_a_linterieur(point, figure):
            return True
        else:
            return False
            
""" 
Test

figure1 = [(0, 0), (0, 1), (1, 1), (1, 0)]  # Exemple de première figure sous forme d'une liste de points
figure2 = [(2, 0), (3, 0), (3, 1), (2, 1)]  
figures = [figure1, figure2]  # Liste des figures
point= (0.5,0.5)

print(point_dans_figures(point, figures))  # Appel de la fonction pour obtenir les résultats
"""