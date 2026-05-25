from __future__ import annotations

import io
import os
from datetime import datetime
from typing import List

from PIL import Image
import matplotlib.pyplot as plt
import config as cfg

from src.sim_types import SimulationArtists, SnapshotState



def save_frame(fig: plt.Figure, frames: List[Image.Image]) -> None:
    """
    Cattura lo stato corrente della figura Matplotlib e lo serializza in un buffer di memoria (BytesIO).
    Questa operazione evita costose operazioni di I/O su disco durante l'esecuzione del loop principale,
    mantenendo in RAM una sequenza di oggetti PIL Image.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    frames.append(Image.open(buffer).copy())
    buffer.close()


def finalize_status(artists: SimulationArtists, snapshot_state: SnapshotState, status: str,
                    total_distance: float) -> None:
    """
    Esegue la finalizzazione dell'interfaccia grafica al termine della simulazione.
    In caso di successo (goal_reached), aggiorna il cruscotto con un feedback visivo positivo.
    Registra inoltre l'ultimo frame prima della terminazione del ciclo.
    """
    if status == "goal_reached":
        artists.stats_text.set_text(f" GOAL RAGGIUNTO! Distanza Finale Percorsa: {total_distance:.2f} ")
        artists.stats_text.set_color("green")

    save_frame(artists.fig, snapshot_state.frames)


def write_gif(media_dir: str, frames: List[Image.Image]) -> None:
    """
    Gestisce l'esportazione sequenziale dei frame bufferizzati in formato GIF animata.
    Implementa un sistema di numerazione incrementale tramite file di log (prova_counter.txt)
    per mantenere lo storico delle sessioni di test effettuate.
    """
    # 1. Recupero del contatore di sessione o inizializzazione a 1
    counter_file = "prova_counter.txt"
    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            try:
                i = int(f.read().strip())
            except ValueError:
                i = 1
    else:
        i = 1

    # Aggiornamento e persistenza del contatore per la sessione successiva
    with open(counter_file, "w") as f:
        f.write(str(i + 1))

    # 2. Definizione del filename con i parametri di configurazione attuali
    # Inclusione di: Seed, Sensing Range, Raggio Robot, Numero raggi LIDAR
    filename = f"Prova_{i}_TB_Seed{cfg.SEED}_SR{cfg.SENSING_RANGE}_Radius{cfg.ROBOT_RADIUS}_Rays{cfg.LIDAR_NUM_RAYS}.gif"

    gif_path = os.path.join(media_dir, filename)

    # 3. Serializzazione della sequenza di immagini in formato GIF animata
    # optimize=True riduce la dimensione del file; duration=300ms imposta il frame rate.
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=300,
        loop=0
    )
