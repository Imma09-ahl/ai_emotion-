@echo off
chcp 65001 >nul
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║      EMOTION AI — Démarrage du serveur       ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo  Chargement du modèle d'IA en cours...
echo  (Cela peut prendre quelques secondes)
echo.
echo  Une fois démarré, ouvrez votre navigateur sur :
echo  → http://127.0.0.1:8000
echo.
echo  Appuyez sur CTRL+C pour arrêter le serveur.
echo  ─────────────────────────────────────────────────
echo.

cd /d %~dp0
.\.venv\Scripts\python.exe web\server.py
pause

