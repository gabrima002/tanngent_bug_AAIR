from __future__ import annotations

import io
import os
import random
import matplotlib.pyplot as plt
import config as cfg

from src.robot import Robot
from src.sim_playback import finalize_status, save_frame, write_gif
from src.sim_render import (
    current_heuristic_from_robot,
    setup_figure,
    update_robot_artists,
    update_snapshot_artists,
    update_stats_text,
)
from src.sim_types import SnapshotState
from src.worldgen import create_environment


def run_simulation() -> None:
    """
    Questa funzione gestisce l'intero ciclo di vita della simulazione:
    1. Inizializzazione dell'ambiente (mappa, ostacoli, start, goal).
    2. Istanziazione dell'agente (Robot) e dell'interfaccia grafica (Matplotlib).
    3. Esecuzione del loop temporale discreto (percezione, decisione, azione).
    4. Aggiornamento del rendering in tempo reale.
    5. Esportazione del risultato finale sotto forma di animazione (GIF).
    """

    # ==========================================
    # FASE 1: SETUP DELL'AMBIENTE E I/O
    # ==========================================

    # Creazione della directory per il salvataggio dei media generati dalla simulazione
    media_dir = "media"
    os.makedirs(media_dir, exist_ok=True)
    print(f"La cartella di salvataggio è: {os.path.abspath(media_dir)}")

    # Pausa in secondi tra un frame e l'altro per il rendering in tempo reale.
    # Un valore basso (es. 0.01) rende l'animazione a schermo fluida e rapida.
    realtime_pause = 0.01

    # Fissaggio del seed per il generatore pseudo-casuale.
    # Garantisce che la stessa mappa venga generata a ogni avvio se il seed non cambia,
    # permettendo il debug di specifiche configurazioni di ostacoli.
    if cfg.SEED is not None:
        random.seed(cfg.SEED)

    # Generazione procedurale dell'ambiente fisico (ostacoli, confini, punti di interesse)
    environment = create_environment()

    # Istanziazione del robot con le specifiche fisiche e sensoriali definite nel config
    robot = Robot(environment.start, environment.goal, cfg.SENSING_RANGE, robot_radius=cfg.ROBOT_RADIUS)

    # Inizializzazione della figura Matplotlib e di tutti gli "artists" (oggetti grafici
    # manipolabili come linee, poligoni e testi) che comporranno la UI.
    artists = setup_figure(environment, robot)

    # Struttura dati per memorizzare i frame per la GIF e la cronologia dell'euristica
    snapshot_state = SnapshotState()

    # Variabili di tracciamento dello stato globale
    status = "running"
    total_distance = 0.0

    # ==========================================
    # FASE 2: LOOP DI SIMULAZIONE (TIME STEPS)
    # ==========================================

    for step in range(cfg.MAX_STEPS):
        # Graceful exit: interrompe il ciclo se l'utente chiude manualmente la finestra del grafico
        if not plt.fignum_exists(artists.fig.number):
            break

        # Memorizzazione della posizione pre-movimento per calcolare delta vettoriali e grafiche
        old_position = robot.position

        # --- FASE 2.1: PERCEZIONE (SENSE) ---
        # Il robot interroga l'ambiente circostante simulando un sensore LIDAR a X° in base a quanto impostato in config.py.
        # dap: Distances and Angles Profile (array di distanze lette).
        # dps: Discontinuity Points (spigoli/angoli individuati dal sensore).
        dap, dps = robot.sense_environment(environment.obstacles)

        # --- FASE 2.2: VALUTAZIONE EURISTICA (THINK) ---
        # Recupera il miglior valore euristico storico e calcola quello attuale basato
        # sulla nuova visuale, fondamentale per rilevare i minimi locali.
        previous_heuristic = snapshot_state.last_heuristic
        current_heuristic = current_heuristic_from_robot(robot, dap, dps)

        # --- FASE 2.3: AZIONE (ACT) ---
        # La State Machine del robot esegue il calcolo cinematico decidendo se muoversi
        # verso il goal o seguire un ostacolo (Boundary Following).
        # Aggiorna internamente la propria posizione e restituisce lo stato (running, goal_reached, stuck).
        status = robot.step(environment.obstacles)

        # Aggiornamento della memoria globale con la nuova euristica valutata
        snapshot_state.last_heuristic = current_heuristic

        # ==========================================
        # FASE 3: RENDERING E SALVATAGGIO GRAFICO
        # ==========================================

        # Aggiorna la posizione e l'orientamento del modello fisico del robot sulla mappa
        update_robot_artists(artists, robot)

        # Aggiorna le componenti visive analitiche (raggi LIDAR, linee di mira d_reach, marker di discontinuità)
        update_snapshot_artists(artists, snapshot_state, robot, old_position, dap, dps)

        # Cattura lo stato attuale della figura Matplotlib e lo converte in un'immagine PIL
        # aggiungendolo all'array di frame per la successiva esportazione GIF.
        save_frame(artists.fig, snapshot_state.frames)

        # Aggiorna il pannello testuale inferiore con le metriche calcolate (distanze, euristiche, numero di step)
        total_distance = update_stats_text(artists, robot, step, current_heuristic, previous_heuristic)

        # Segnala al backend di Matplotlib che i dati grafici sono cambiati e forza il ridisegno immediato
        artists.fig.canvas.draw_idle()
        artists.fig.canvas.flush_events()

        # Mette in pausa l'esecuzione del thread per permettere al motore grafico di renderizzare il frame a schermo
        plt.pause(realtime_pause)

        # ==========================================
        # FASE 4: CONDIZIONI DI TERMINAZIONE
        # ==========================================

        if status in ("goal_reached", "stuck"):
            # Aggiorna la UI per mostrare il messaggio di completamento o fallimento
            finalize_status(artists, snapshot_state, status, total_distance)
            break

    # ==========================================
    # FASE 5: POST-ELABORAZIONE E TEARDOWN
    # ==========================================

    # Esporta l'animazione solo se il robot è arrivato con successo al target
    # e se sono stati effettivamente catturati dei frame.
    if snapshot_state.frames and status == "goal_reached":
        write_gif(media_dir, snapshot_state.frames)

    # Disattiva la modalità interattiva di Matplotlib.
    # Questo impedisce alla finestra di chiudersi automaticamente al termine dello script,
    # permettendo all'utente di ispezionare il percorso finale.
    plt.ioff()
    plt.show()
