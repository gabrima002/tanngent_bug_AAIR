# Simulazione Navigazione Autonoma: Tangent Bug

Questo progetto implementa un ambiente di simulazione 2D per l'algoritmo di navigazione **Tangent Bug**. Il sistema modella un robot dotato di un sensore (simile a un LIDAR a 360°) incaricato di raggiungere un obiettivo (Goal) navigando in un ambiente pieno di ostacoli generati casualmente.

Tutti i parametri della simulazione (grandezza del robot, raggio del sensore, numero di ostacoli, velocità, ecc.) sono **completamente configurabili** in modo semplice tramite il file `config.py`.

## Caratteristiche Principali

* **Logica a Due Stati:** Il robot alterna in modo autonomo due comportamenti principali per trovare la strada:
  * *Move-to-Goal*: Il robot si muove direttamente verso il traguardo o verso lo spigolo visibile che lo avvicina di più.
  * *Boundary-Following*: Se il robot finisce in un vicolo cieco (minimo locale), inizia a seguire il perimetro dell'ostacolo per aggirarlo.
* **Mappe Casuali ma Sicure:** La mappa viene generata dinamicamente ad ogni avvio. Gli ostacoli vengono posizionati in modo casuale ma intelligente, assicurando che non si sovrappongano al punto di partenza o al traguardo.
* **Sensore Simulato:** Il robot calcola le distanze dagli ostacoli in tempo reale e individua gli spigoli liberi per capire dove può passare.
* **Grafica in Tempo Reale:** L'interfaccia mostra la mappa 2D, la visuale del sensore e un pannello con tutti i dati in tempo reale (distanze, stato attuale del robot).
* **Esportazione Automatica:** Alla conclusione della prova con successo, il sistema salva automaticamente una GIF animata della traiettoria nella cartella `media/`.

## Pseudo-codice dell'Algoritmo

Il comportamento del robot si basa su queste regole formali:

**1. Repeat** behavior 1: move-to-goal
* a. compute discontinuity points $O_k$
* b. move to goal, if reachable, or to the discontinuity point with minimal heuristic distance $h_k(q)$

**until**
* a. goal is reached, or
* b. minimal heuristic distance $h_k(q)$ begins to increase

**2. Repeat** behavior 2: boundary following
* a. compute discontinuity points $O_k$ and distances $d_{\text{reach}}$, $d_{\text{followed}}$
* b. follow boundary continuing in same direction as before

**until**
* a. goal is reached, or
* b. a complete cycle is performed (goal is unreachable), or
* c. $d_{\text{reach}} < d_{\text{followed}}$

## Struttura del Progetto

* `main.py`: Il file da avviare per far partire il programma.
* `config.py`: Il pannello di controllo dove puoi modificare tutte le impostazioni e i parametri del mondo e del robot.
* `simulation.py`: Gestisce il ciclo di vita della simulazione (muove il robot e aggiorna lo schermo).
* `robot.py`: Il "cervello" del robot e le sue regole di movimento.
* `worldgen.py`: Si occupa di creare la mappa e disegnare gli ostacoli.
* `geometry.py`: Contiene i calcoli matematici e verifica se il robot sbatte contro i muri.
* `sim_render.py` & `sim_playback.py`: Gestiscono la grafica a schermo e il salvataggio dell'animazione GIF.
* `sim_types.py`: Contenitori di dati per tenere traccia in modo ordinato di cosa succede nella simulazione.

## Installazione

Si consiglia di utilizzare un ambiente virtuale (come `venv` o `conda`). Per installare le librerie necessarie, apri il terminale ed esegui:

```bash
pip install -r requirements.txt

