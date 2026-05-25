# Simulazione Navigazione Autonoma: Tangent Bug

Questo progetto implementa un ambiente di simulazione 2D per l'algoritmo di navigazione **Tangent Bug**. Il sistema modella un robot dotato di un sensore (simile a un LIDAR) incaricato di raggiungere un obiettivo (Goal) navigando in un ambiente pieno di ostacoli generati casualmente.

Tutti i parametri della simulazione (grandezza del robot, raggio del sensore, numero di ostacoli, velocità, ecc.) sono **completamente configurabili** in modo semplice tramite il file `config.py`.

## Caratteristiche Principali

* **Logica a Due Stati:** Il robot alterna in modo autonomo due comportamenti principali per trovare la strada:
  * *Move-to-Goal*: Il robot si muove direttamente verso il traguardo o verso lo spigolo visibile che lo avvicina di più.
  * *Boundary-Following*: Se il robot finisce in un vicolo cieco (minimo locale), inizia a seguire il perimetro dell'ostacolo per aggirarlo.
* **Mappe Casuali ma Sicure:** La mappa viene generata dinamicamente ad ogni avvio. Gli ostacoli vengono posizionati in modo casuale ma intelligente, assicurando che non si sovrappongano al punto di partenza o al traguardo.
* **Sensore Simulato:** Il robot calcola le distanze dagli ostacoli in tempo reale e individua gli spigoli liberi per capire dove può passare.
* **Grafica in Tempo Reale:** L'interfaccia mostra la mappa 2D, la visuale del sensore e un pannello con tutti i dati in tempo reale (distanze, stato attuale del robot).
* **Esportazione Automatica:** Alla conclusione della prova con successo, il sistema salva automaticamente una GIF animata della traiettoria nella cartella `media/`.

# Pseudo-codice dell'Algoritmo

## Ripetere il comportamento 1: movimento verso il goal

1. Calcolare i punti di discontinuità ((O_k)).
2. Muoversi verso il goal, se raggiungibile; altrimenti dirigersi verso il punto di discontinuità con distanza euristica minima ((h_k(q))).

**Finché non si verifica una delle seguenti condizioni:**

* il goal viene raggiunto; oppure
* la distanza euristica minima ((h_k(q))) inizia ad aumentare.

---

## Ripetere il comportamento 2: inseguimento del bordo

1. Calcolare i punti di discontinuità ((O_k)) e le distanze ((d_{\text{reach}})) e ((d_{\text{followed}})).
2. Seguire il bordo mantenendo la stessa direzione di percorrenza adottata in precedenza.

**Finché non si verifica una delle seguenti condizioni:**

* il goal viene raggiunto; oppure
* viene completato un ciclo completo (il goal è irraggiungibile); oppure
* ((d_{\text{reach}} < d_{\text{followed}})).


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

