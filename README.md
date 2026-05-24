# Simulazione Navigazione Autonoma: Tangent Bug

Questo progetto implementa un ambiente di simulazione 2D per l'algoritmo di navigazione reattiva **Tangent Bug**. Il sistema modella un agente autonomo (robot) equipaggiato con un sensore LIDAR simulato a 360°, incaricato di raggiungere un obiettivo (Goal) navigando in un ambiente con ostacoli generati proceduralmente.

## Caratteristiche Principali

* **Macchina a Stati Finiti (FSM):** Il robot alterna in modo autonomo due comportamenti principali:
  * *Motion-to-Goal*: Navigazione a gradiente basata sull'euristica visiva locale.
  * *Boundary-Following*: Aggiramento del perimetro degli ostacoli per uscire dai minimi locali (trappole spaziali).
* **Generazione Procedurale:** La mappa (ostacoli concavi, convessi e poligonali) viene generata dinamicamente ad ogni avvio tramite tecniche di Reject Sampling, garantendo sempre la presenza di un percorso valido.
* **Sensore LIDAR Simulato:** Raycasting analitico per l'individuazione di ostacoli e punti di discontinuità (jump edges).
* **Rendering in Tempo Reale:** L'interfaccia, sviluppata con `matplotlib`, mostra la mappa 2D globale, il grafo di tangenza locale 1D (Local Tangent Graph) e un cruscotto telemetrico con metriche di navigazione (distanze ed euristiche).
* **Esportazione Automatica:** Alla conclusione della prova con successo, il sistema salva automaticamente una GIF animata della traiettoria nella cartella `media/`.

## Struttura del Progetto

* `main.py`: Entry point dell'applicazione.
* `config.py`: Parametri centralizzati (dimensioni mappa, tolleranze, setup sensori).
* `simulation.py`: Orchestratore principale del loop temporale (percezione-decisione-azione).
* `robot.py`: Logica di controllo, FSM e calcolo cinematico dell'agente.
* `worldgen.py`: Factory procedurale per la generazione della mappa e degli ostacoli.
* `geometry.py`: Primitive matematiche, algebra vettoriale e algoritmi di collision detection (Narrow & Broad phase).
* `sim_render.py` & `sim_playback.py`: Moduli per il rendering Matplotlib e I/O per il salvataggio GIF.
* `sim_types.py`: Strutture dati (Data Classes) per l'incapsulamento dello stato di simulazione.

## Installazione

Si consiglia di utilizzare un ambiente virtuale (`venv` o `conda`). Per installare le dipendenze necessarie, esegui:

```bash
pip install -r requirements.txt

