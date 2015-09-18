Projet Game Of Life EISTI I3 2015.
Auteur : Micha�l Roynard.
Environnement de d�veloppement : Microsoft Visual Studio 2013. Nvidia Cuda Toolkit 6.5-7.5 (configur� pour 7.5 actuellement)

-------------------------
Procecus de build :
1) regarder quel est la version de Nvidia Cuda toolkit est install� sur l'ordinateur :

2) Se rendre dans : C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\BuildCustomizations
Il doit y avoir des fichiers de type :
CUDA 6.5.props
CUDA 7.0.props
CUDA 7.5.props
Prendre le plus r�cent install� sur votre ordinateur.

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
- GameOfLife : aucun probl�me
- GameOfLife_Cuda aller dans les propri�t�s du projet :
	1) s�lectionner All configuration + x64 puis :
		configuration properties > debugging > environnement > mettre PATH=$(FREEGLUT_BIN_DIR_X64);%PATH%
	2) s�lectionner All configuration + win32 puis :
		configuration properties > debugging > environnement > mettre PATH=$(FREEGLUT_BIN_DIR_WIN32);%PATH%
- Appli1-Julia : exactement la m�me chose.

Pour une ex�cution en dehors de visual, il faudra copier les dll de freeglut avec l'executable.
Pour un d�buggage nsight depuis visual, il faudra recopier aussi les dll freeglut (il n'utilises pas le path de visual).

-----------------------------
Machine pour le bench :
CPU i7-2660k 3.40Ghz
GPU NVidia GeForce GTX 680

-----------------------------

Traitement du sujet :

----

1) : Impl�menter GameOfLife version CPU

Question trait�e dans le projet GameOfLife. Voir le main.cpp pour la configuration des bench et les preuves d'impl�mentation.
NOTE : 3 types d'ex�cutions sont disponibles :
- version s�rial : aucune optimisation, aucune parall�lisation
- version omp : parall�lisation openMP lors du calcul de la g�n�ration suivante
- version std::thread : essaie d'impl�mentation parrall�le avec std::thread (cette impl�mentation sort de mauvais r�sultats avec de tr�s petites grilles. Certainement qu'il faudrait ne pas respawn le pool de thread � chaque g�n�ration.)

RESULTATS :
Avec les configurations propos�es (1000 g�n�rations sur une grille de taille 2^8 = 256)
Serial : 4076ms
OpenMP : 2001ms (facteur d'acc�l�ration : 2.036)
std::thread : 2104ms (facteur d'acc�l�ration : 1.973)

Les r�sultats sont plut�ts logiques : la parall�lisation CPU fait gagner du temps.

----

2-4) : Impl�menter 2 kernels cuda, un utilisant la m�moire globale, un utilisant la m�moire shared pour faire la m�me simulation.

Question trait�e dans le projet GameOfLife_Cuda. Voir le main.cpp pour la configuration (mettre le bool openGL � false pour lancer les kernels en console).

RESULTATS :
Avec les configurations propos�es (1000 g�n�rations sur une grille de taille 2^8 = 256)
Kernel global memory : 349ms (facteur d'acc�l�ration 11.67 par rapport � CPU serial, 5.734 par rapport � CPU openMP)
Kernel shared memory : 22ms (facteur d'acc�l�ration 15.86 par rapport au kernel global memory, 185.3 par rapport � CPU serial, 90.90 par rapport � CPU openMP)

Le fait que l'impl�mentation GPU soit faite en C++ avec des indirections et des structures de donn�e n'aide pas � promouvoir sa performance.
Le facteur d'acc�l�ration shared -> global memory est r�ellement int�ressant et r�compense vraiment les efforts pass� � mettre au point ce kernel.


----

5) : Afficher avec OpenGL le r�sultat sorti par les kernels cuda.

Question trait�e dans le projet GameOfLife_Cuda. Il faut mettre le bool openGL � true dans le main.cpp puis aller dans Display.cpp pour la configuration :
Il existe 2 modes :
- Interop (OpenGL va directement lire dans les buffer de la carte graphique le r�sultat des kernels cuda)
- GPU (OpenGL r�cup�re sur le CPU gr�ce � cudaMemcopy deviceToHost le calcul du kernel puis r�injecte les infos dans la CG)
Pour changer de mode il suffit de mettre le bool interop_ � true ou false dans Display.cpp .

La fen�tre est redimentionnable qu'en mode interop (logique).
Ca marche !