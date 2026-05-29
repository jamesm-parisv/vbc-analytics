# VBC Analytics — version recruteur

Version publique courte du rapport Streamlit **VBC Analytics**.

Cette version met l’accent sur le storytelling recruteur : texte court, conclusions fortes et graphiques compacts directement à côté des insights.

Cette version est pensée pour un portfolio Data Analyst : elle conserve la page de rapport, les insights visuels, les graphiques clés, les limites statistiques et un emplacement pour le lien GitHub.

## Contenu

- **Rapport** : contexte VBC, objectif BMAB, questions analytiques, KPIs et graphiques de synthèse.
- **Insights** : quatre conclusions principales, chacune accompagnée d’un graphique compact pour renforcer le storytelling :
  - donut des matchs gagnés par le joueur avec le meilleur PR ;
  - PR moyen par ronde et PR vs température de Paris ;
  - heatmap catégorie joueur vs catégorie adversaire ;
  - répartition des catégories PR par tournoi.
- **Graphiques clés** : versions détaillées des graphiques des insights, plus des vues complémentaires : écart de PR, longueur du match, progression par catégorie, part des joueurs M3 ou mieux, PR moyen global par tournoi.
- **Limites & code** : taille d’échantillon, matchs courts, variance du backgammon, biais de sélection, effet des joueurs réguliers et emplacement pour le lien GitHub.

## Ajouter le lien GitHub

Dans `app.py`, remplacer :

```python
GITHUB_URL = ""
```

par exemple par :

```python
GITHUB_URL = "https://github.com/jamesmacnaughtan/vbc-analytics"
```

## Lancer localement

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Sur Windows PowerShell, depuis le dossier du projet :

```powershell
python -m streamlit run app.py
```

## Déploiement

1. Créer un dépôt GitHub public.
2. Ajouter le contenu de ce dossier à la racine du dépôt.
3. Déployer sur Streamlit Community Cloud avec `app.py` comme fichier principal.
