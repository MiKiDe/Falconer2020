# Script oral

P : 1,2,3,8,10

S : 4,5,6,7,9

# Slide 1

On va donc vous présenter un bilan de ce mi-stage. On a travaillé sur un théorème de Falconer et sa réalisation encadré par Valentin De Bortoli et Agnes Desolneux.



## Slide 2

Le théorème principal de l’aricle de Falconer que nous avons eu à étudir est le suivant : (...)

On voit qu’on a différentes difficultés théoriques pour des élèves de L3 : U_nk provient des variétés fibrées, M_nk est un mesure sur les Grassmaniennes. On a donc dû se renseigner aussi en géométrie différentielle et en théorie de la mesure sur des espaces un peu atypiques.

Mais dans le mode réel, le théorème est le suivant : étant donné un ensemble de formes 2D que l’on souhaite projeter, il existe un objet 3D qui permet d’obtenir ces images par projection (en faisant passer de la lumière par exemple).

## Slide 3

Pour l’instant, nous avons étudié la plupart de l’article de Falconer. On a d’abord étudié une construction en dimension 1 dont l’énoncé est ici. Étant donné un segment I, on peut toujours construire un ensemble de segments dont la projection est quasiment celle de I pour un petit intervalle d’angles. On peut choisir l’épaisseur de cette construction (rho), la largeur de l’intervalle d’angles (alpha) et l’erreur sur la projection (epsilon).

## Slide 4

On commence par la construction élémentaire suivante (rappeler l’utilité du camembert)

* On va récursivement remplacer chaque segment par m segments faisant chacun un angle alpha avec le segment sur lequel la transformation est appliquée. On obtient la figure suivant pour m=2 et s =5 où s est le nombre d’applications. 
* En choisissant judicieusement s, on vérifiera alors les propriétés précédentes.

+ expliquer vite fait les figures
## Slide 5

Je le ferai

INSISTER QUE C’EST UNE INNITIATIVE DE NOTRE PART

Question : la lire, expliquer brièvement de quoi il en retourne.

Points importants :

* Système de fonctions itérées
* Mesure = prop géométrique
* Difficulté pour les autres constructions, toujours en état de recherche.

## Slide 6

Ensuite, on va remplacer tous les segments par une multitude de segments parallèles cf. fig...Cela va nous permettre de jouer sur les intervalles des angles, mais on perd un petit delta au niveau de l’intervalle blanc.

## Slide 7

On va ensuite appliquer une transformation affine pour jouer sur les intervalles : permet d'obtenir quasiment tout les angles (à delta près).

(rappeler en revenant sur la slide précédente que chaque segment vert est en fait une union de segments parallèles).

## Slide 8

Puis, en réappliquant toute la construction sur chaque segment, et en symétrisant les constructions par rapport à chaque segment, on obtient un ensemble beaucoup plus complexe mais avec des intervalles d’angles symétriques qui correspondent au théorème annoncé au début.

INSISTER QUON A TOUT CONSTRUIT EN PYTHON, c’est une initiative de notre part

## Slide 9

Cas de la 2D fini : maintenant généralisation à la dimension n avec les projections sur des hyperplans. 
* présenter (n-1)-intervalles

Etant donné I un (n-1)-intervalle, il existe un ensemble E union finie de (n-1)-intervalles dont la projection sur les hyperplans dont l'orthogonale est proche de (x1) contient celle de I
et dont la projection sur "presque tout" les autres hyperplans est petite. 
Encore une fois on peut faire varier la distance de E à I (rho), les plages d'angle (alpha et delta) et la taille de la projection de E (epsilon).

* Pour quantifier la distance de pi ortho à (x1): on utilise les theta_r.



## Slide 10

th précédent : condition portant sur les theta_r(pi) qui est pas toujours défini. Pour que le théorème ait un intérêt il faut que ca soit défini pour la plupart des pi. or iff blabla
* introduction de Gn,k car structure de variété lisse -> parties néglibeables naturelles. 
* Muni de ces defs on a montré la prop qui est cqfd

## Slide 11

Soit je conclus tout seul, soit on fait à deux genre moit moit.

