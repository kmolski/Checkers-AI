import logging

from training import BaseTrainingSession
from training import TournamentSession

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO
)

# BASE TRAINING

BaseTrainingSession.train()

# TOURNAMENTS

TournamentSession.train()
