Projet Game Of Life EISTI I3 2015.
Auteur : Michaël Roynard.
Environnement de développement : Microsoft Visual Studio 2013. Nvidia Cuda Toolkit 6.5-7.5 (configuré pour 7.5 actuellement)

-------------------------
Procecus de build :
1) regarder quel est la version de Nvidia Cuda toolkit est installé sur l'ordinateur :

2) Se rendre dans : C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\BuildCustomizations
Il doit y avoir des fichiers de type :
CUDA 6.5.props
CUDA 7.0.props
CUDA 7.5.props
Prendre le plus récent installé sur votre ordinateur.

3)Ensuite, il faut ouvrir avec notepad les fichiers suivants :
 GameOfLife_Cuda/GameOfLife_Cuda.vcxproj
 Appli1-Julia/Appli1-Julia.vcxproj
4) Chercher et remplacer les lignes suivantes dans les 2 fichiers :
	<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 7.5.props" />
	...
	<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 7.5.targets" />
	
	IL FAUT METTRE LE NUMERO DE VERSION D'UN FICHIER PROPS QUI EXISTE SUR VOTRE ORDINATEUR.

5) Uniquement maintenant : ouvrir le fichier solution avec GameOfLife.sln avec Visual Studio 2013.
Si le fichier .props n'est pas bon dans le .vcxproj, visual refusera de charger le projet.

6) Build :
Aller dans menu > build > batch build : select all + rebuild

7) Execution :
- GameOfLife : aucun problème
- GameOfLife_Cuda aller dans les propriétés du projet :
	1) sélectionner All configuration + x64 puis :
		configuration properties > debugging > environnement > mettre PATH=$(FREEGLUT_BIN_DIR_X64);%PATH%
	2) sélectionner All configuration + win32 puis :
		configuration properties > debugging > environnement > mettre PATH=$(FREEGLUT_BIN_DIR_WIN32);%PATH%
- Appli1-Julia : exactement la même chose.

Pour une exécution en dehors de visual, il faudra copier les dll de freeglut avec l'executable.
Pour un débuggage nsight depuis visual, il faudra recopier aussi les dll freeglut (il n'utilises pas le path de visual).

-----------------------------
Machine pour le bench :
CPU i7-2660k 3.40Ghz
GPU NVidia GeForce GTX 680

-----------------------------

Traitement du sujet :

----

1) : Implémenter GameOfLife version CPU

Question traitée dans le projet GameOfLife. Voir le main.cpp pour la configuration des bench et les preuves d'implémentation.
NOTE : 3 types d'exécutions sont disponibles :
- version sérial : aucune optimisation, aucune parallélisation
- version omp : parallélisation openMP lors du calcul de la génération suivante
- version std::thread : essaie d'implémentation parrallèle avec std::thread (cette implémentation sort de mauvais résultats avec de très petites grilles. Certainement qu'il faudrait ne pas respawn le pool de thread à chaque génération.)

RESULTATS :
Avec les configurations proposées (1000 générations sur une grille de taille 2^8 = 256)
Serial : 4076ms
OpenMP : 2001ms (facteur d'accélération : 2.036)
std::thread : 2104ms (facteur d'accélération : 1.973)

Les résultats sont plutôts logiques : la parallélisation CPU fait gagner du temps.

----

2-4) : Implémenter 2 kernels cuda, un utilisant la mémoire globale, un utilisant la mémoire shared pour faire la même simulation.

Question traitée dans le projet GameOfLife_Cuda. Voir le main.cpp pour la configuration (mettre le bool openGL à false pour lancer les kernels en console).

RESULTATS :
Avec les configurations proposées (1000 générations sur une grille de taille 2^8 = 256)
Kernel global memory : 349ms (facteur d'accélération 11.67 par rapport à CPU serial, 5.734 par rapport à CPU openMP)
Kernel shared memory : 22ms (facteur d'accélération 15.86 par rapport au kernel global memory, 185.3 par rapport à CPU serial, 90.90 par rapport à CPU openMP)

Le fait que l'implémentation GPU soit faite en C++ avec des indirections et des structures de donnée n'aide pas à promouvoir sa performance.
Le facteur d'accélération shared -> global memory est réellement intéressant et récompense vraiment les efforts passé à mettre au point ce kernel.


----

5) : Afficher avec OpenGL le résultat sorti par les kernels cuda.

Question traitée dans le projet GameOfLife_Cuda. Il faut mettre le bool openGL à true dans le main.cpp puis aller dans Display.cpp pour la configuration :
Il existe 2 modes :
- Interop (OpenGL va directement lire dans les buffer de la carte graphique le résultat des kernels cuda)
- GPU (OpenGL récupère sur le CPU grâce à cudaMemcopy deviceToHost le calcul du kernel puis réinjecte les infos dans la CG)
Pour changer de mode il suffit de mettre le bool interop_ à true ou false dans Display.cpp .

La fenêtre est redimentionnable qu'en mode interop (logique).
Ca marche !