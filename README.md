

# Tangent Bug Algorithm: Implementation and Lidar Simulation

Questo repository contiene un'implementazione in Python dell'algoritmo di navigazione Tangent Bug, applicato a un robot puntiforme che opera in ambienti 2D complessi. Il progetto include una simulazione realistica di un sensore Lidar a risoluzione finita e un'analisi comparativa delle prestazioni al variare dei parametri cinematici e di scansione.

## Descrizione del Progetto

L'obiettivo è guidare un agente robotico verso un target evitando ostacoli poligonali (anche non convessi) utilizzando la libreria shapely per la gestione delle collisioni e della geometria.

Sia $q$ la posizione attuale del robot e $q_{goal}$ la posizione del target. Definiamo $V$ l'insieme dei punti rilevati dal sensore entro il raggio $R$. Per ogni punto di continuità $O_i$ rilevato sull'ostacolo, la distanza stimata verso il goal attraverso $O_i$ è definita come:

$$
d(q, q_{goal}) = \|q - O_i\| + \|O_i - q_{goal}\|
$$

Il robot sceglie la direzione che minimizza questa funzione:

$$
i^* = \arg\min_{i} \left( \|q - O_i\| + \|O_i - q_{goal}\| \right)
$$

L'algoritmo prevede l'alternarsi di due comportamenti:

 1. Comportamento 1 (Motion-to-Goal)
Il robot si muove in linea retta verso il punto $O_{i^*}$ che minimizza l'euristica, a patto che:

$$
d_{min}(t) < d_{reached}
$$

dove $d_{min}$ è la distanza minima mai registrata tra l'ostacolo e il goal, e $d_{reached}$ è la distanza attuale.

 2. Comportamento 2 (Boundary Following)
Viene attivato quando il robot rileva un minimo locale della funzione distanza. In questa fase, il robot segue il bordo dell'ostacolo mantenendo una distanza di sicurezza, finché non viene soddisfatta la condizione di distacco:

$$
d_{leave} < d_{min}
$$

dove $d_{leave}$ è la distanza tra il punto di uscita potenziale e il goal.

---

### Caratteristiche principali

* Sensore Lidar realistico: Simulazione di un raggio di visione ($R$) e di una risoluzione angolare finita ($\Delta\theta$).
* Gestione Ostacoli: Supporto per geometrie complesse e non convesse (Libreria shapely).
* Visualizzazione Dinamica: Generazione automatica di GIF/MP4 che mostrano il raggio d'azione del sensore, i punti di discontinuità rilevati e lo stato del comportamento attuale.

---

## Requisiti e Installazione

Il progetto richiede Python 3.x e le seguenti librerie:

* shapely
* numpy
* matplotlib

