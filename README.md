# Adversarial-Deep-Learning-For-Open-Set-Classification
## Partie GAN:
PTS8 contient 2 GAN (couches denses/convolutions) qui permettent de générer des images à partir de MNIST. 
## Création de Base de données pour le Rejecteur: 
Après avoir enregistré les images générées par les GAN avec les probabilités associées du discriminateur, on choisit à un seuil fixé les bonnes images (ceux qui seront après rejeter par le rejecteur). 
## Le Rejecteur:
Nous avons opté pour tester deux types de rejecteurs: 
####      Un avec une seule sortie:
          C'est une régression qui permet d'obtenir une probabilité en sortie.
####      Un avec deux sorties:
          C'est une classification dans deux classes (0: à garder et 1: à rejeter)
##  Le classifieur: 
Dans ce modèle notre classifieur est entrainé sur toute la base d'entrainement de MNIST (les 10 classes). 
## Le teste: 
A la sortie de notre classifieur on fait un test sur l'accuracy sur la base de test de MNIST (qui n'a jamais été utilisée pour entrainer n'importe quel modèle). 
L'objectif souhaité est d'obtenir une accuracy de 1 en sortie avec notre rejecteur. 
Cependant, ce n'était pas le cas. 
#### Nous avons alors regardé les images rejetées et leurs classifications: 
Notre classifieur sans utiliser de rejecteur a une accuracy de 99,49% sur 10 000 exemples de test, soit 51 images mal-classées.
Nous avons regardé alors le nombre d'images rejetées et la prédiction de notre classifieur sur ces images: 
##### Le rejecteur à une sortie: 
Il a rejeté 146 images sur 10 000, dont 5 seront mal-classées une fois passées par le classifieur. Soit un taux de réussite de 9,8% sur les images mal-classées. 
NB: Le nombre d'images rejetées qui sont classifiables est très élevé. Ceci peut être expliqué par un seuil pas assez bas choisi pour le discriminateur. En effet On prend peut être de bons images (classifiables) et on les considère comme mauvaises.
##### Le rejecteur à une sortie: 
Il a rejeté 198 images sur 10 000, dont 7 seront mal-classées une fois passées par le classifieur. Soit un taux de réussite de 13,7% sur les images mal-classées. 
NB: Le nombre d'images rejetées qui sont classifiables est PLUS élevé que le premier rejecteur. 


## Propositions de solutions: 
#### Faire plusieurs mise aléatoires des données (Chaffle) au lieu d'une seule:
La base de données du rejecteur peut être biaisé par un ordre aléatoire. Soit la base n'était pas assez mélangée. 
On peut aussi changer la graine (Seed) à chaque mise aléatoire de données.

--> Solution Pas encore implémentée.

#### Passer d'un rejecteur à une sortie à un rejcteur à deux sorties: 
En effet en utilisant un rejecteur à une sortie, nous sommes amenés à choisir un seuil fixe suivant lequel on considère une image comme une bonne image (à garder) ou une mauvaise (à rejeter). 
Dans un rejecteur à deux sorties c'est l'algorithme qui impose ce seuil. 

--> Solution implémentée dans notre algorithme mais son efficacité est à en discuter. 

#### Ajouter un initialiseur dans le rejecteur à une sortie:
kernel_initializer=initializers.RandomNormal(stddev=0.02) en conv2D

--> Solution Pas encore implémentée.

#### Augmenter le nombre de données: 
Rajouter d'autres GANs. 

--> Solution en cours d'implémentions.

#### Diminuer le seuil du discriminateur: 
Le choix de mauvais exemples et de bons exemples a été réalisé en fixant un seuil dur le discriminateur. 
Ce seuil pourrait être un peu élevé, ce qui amène le rejecteur à considérer des images classifiables comme des images à rejeter.

--> Solution pas encore implémentée.
