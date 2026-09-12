# Fraud Detection App

Application de detection de transactions frauduleuses avec Python, scikit-learn et Streamlit.

Le projet entraine un modele de machine learning sur des transactions financieres, sauvegarde le pipeline de prediction dans un fichier `.pkl`, puis fournit une interface web permettant d'evaluer une transaction saisie manuellement.

> Important: cette application est un outil de demonstration et d'aide a l'analyse. Une prediction ne doit pas etre consideree comme une preuve definitive de fraude.

## Fonctionnalites

- Chargement d'un pipeline de machine learning pre-entraine.
- Saisie interactive des caracteristiques d'une transaction.
- Prise en charge des types `PAYMENT`, `TRANSFER`, `CASH_OUT` et `DEPOSIT`.
- Prediction binaire:
  - `0`: transaction consideree comme non frauduleuse par le modele.
  - `1`: transaction signalee comme potentiellement frauduleuse.
- Affichage immediat du resultat dans l'interface Streamlit.

## Structure du projet

```text
Fraud detection/
|-- fraud_detectionApp.py           # Application Streamlit
|-- Fraud.csv                       # Jeu de donnees des transactions
|-- frauddetection.ipynb             # Exploration, entrainement et analyse
|-- fraud_detection_pipeline.pkl     # Pipeline scikit-learn sauvegarde
|-- README.md                        # Documentation du projet
```

## Jeu de donnees

Le fichier `Fraud.csv` contient des transactions avec les informations suivantes:

| Colonne | Description | Utilisee par l'application |
|---|---|---|
| `step` | Pas de temps de la transaction | Non |
| `type` | Type de transaction | Oui |
| `amount` | Montant de la transaction | Oui |
| `nameOrig` | Identifiant de l'emetteur | Non |
| `oldbalanceOrg` | Solde de l'emetteur avant la transaction | Oui |
| `newbalanceOrig` | Solde de l'emetteur apres la transaction | Oui |
| `nameDest` | Identifiant du destinataire | Non |
| `oldbalanceDest` | Solde du destinataire avant la transaction | Oui |
| `newbalanceDest` | Solde du destinataire apres la transaction | Oui |
| `isFraud` | Cible: transaction frauduleuse ou non | Cible d'entrainement |
| `isFlaggedFraud` | Indicateur de fraude fourni par la source | Non dans l'interface actuelle |

Les colonnes d'identifiant (`nameOrig` et `nameDest`) ainsi que `step` et `isFlaggedFraud` ne sont pas envoyees au modele de production actuel. Le modele utilise donc les six champs suivants:

```text
type
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

## Modele et pipeline

Le fichier `fraud_detection_pipeline.pkl` contient un pipeline scikit-learn compose de deux etapes:

1. **Pretraitement** avec un `ColumnTransformer`:
   - standardisation des variables numeriques avec `StandardScaler`;
   - encodage de la variable categorielle `type` avec `OneHotEncoder(drop="first")`.
2. **Classification** avec une `LogisticRegression` configuree avec:
   - `class_weight="balanced"`, afin de mieux prendre en compte le desequilibre entre les transactions normales et frauduleuses;
   - `max_iter=1000`, pour laisser suffisamment d'iterations a l'optimisation.

Le pipeline est charge au demarrage de l'application:

```python
model = joblib.load("fraud_detection_pipeline.pkl")
```

L'utilisation du pipeline complet est importante: le meme encodage et la meme standardisation qu'a l'entrainement sont automatiquement appliques aux nouvelles donnees.

## Installation

Python 3.9 ou une version plus recente est recommandee.

1. Cloner ou copier le projet.
2. Ouvrir un terminal dans le dossier du projet.
3. Installer les dependances:

```bash
python -m pip install streamlit pandas scikit-learn joblib jupyter
```

Le fichier CSV est volumineux. Il doit etre present localement si vous souhaitez reproduire l'exploration ou l'entrainement du modele.

## Lancer l'application

Depuis le dossier contenant `fraud_detectionApp.py`:

```bash
streamlit run fraud_detectionApp.py
```

Streamlit affiche ensuite une adresse locale, generalement:

```text
http://localhost:8501
```

Dans l'interface:

1. Selectionner le type de transaction.
2. Renseigner le montant.
3. Renseigner les soldes de l'emetteur avant et apres la transaction.
4. Renseigner les soldes du destinataire avant et apres la transaction.
5. Cliquer sur **Predict**.

Le resultat affiche la classe predite et un message indiquant si la transaction est potentiellement frauduleuse.

## Lancer le notebook

Le notebook `frauddetection.ipynb` contient l'espace de travail d'analyse et d'entrainement. Pour l'ouvrir:

```bash
jupyter notebook frauddetection.ipynb
```

ou:

```bash
jupyter lab frauddetection.ipynb
```

Avant de reutiliser le notebook, verifier que:

- le chemin vers `Fraud.csv` est correct;
- les noms de colonnes correspondent au fichier de donnees;
- la cible d'entrainement est `isFraud`;
- le pipeline genere est sauvegarde sous le nom `fraud_detection_pipeline.pkl`;
- les variables envoyees a l'application correspondent exactement aux variables d'entrainement.

## Flux de prediction

```text
Saisie utilisateur
      |
      v
DataFrame pandas avec une transaction
      |
      v
Pipeline scikit-learn sauvegarde
  - standardisation numerique
  - encodage de type
  - regression logistique
      |
      v
Classe predite: 0 ou 1
      |
      v
Message Streamlit
```

## Points de vigilance

- Le chemin du fichier `.pkl` est relatif au dossier depuis lequel Streamlit est lance. Il faut donc executer la commande depuis le dossier du projet, ou adapter le chemin.
- Les valeurs saisies doivent rester realistes et coherentes: par exemple, les soldes apres transaction devraient normalement tenir compte du montant transfere.
- L'application ne montre pas actuellement la probabilite de fraude, le seuil de decision, ni les explications de la prediction.
- Les identifiants de comptes ne sont pas utilises par le modele actuel. Ils pourraient pourtant contenir des informations utiles, a traiter avec prudence pour eviter les fuites de donnees.
- Les performances reelles doivent etre controlees avec des donnees de test separees. Pour un probleme tres desequilibre, l'accuracy seule est insuffisante; il faut aussi examiner la precision, le rappel, le F1-score, la matrice de confusion et, si pertinent, l'aire sous la courbe precision-rappel.
- Le fichier pickle ne doit etre charge que s'il provient d'une source de confiance, car le format pickle peut executer du code lors du chargement.

## Ameliorations possibles

- Ajouter une validation des valeurs saisies et des controles de coherence des soldes.
- Afficher `predict_proba` avec un seuil configurable et une interpretation claire du risque.
- Ajouter des tests automatises pour les colonnes attendues et le format de prediction.
- Comparer plusieurs modeles, par exemple une regression logistique, un Random Forest ou un gradient boosting.
- Utiliser une strategie de validation adaptee au desequilibre des classes.
- Ajouter un suivi des performances dans le temps et une detection du changement de distribution des transactions.
- Conteneuriser l'application et figer les versions des dependances dans un fichier `requirements.txt`.

## Technologies

- Python
- pandas
- scikit-learn
- joblib
- Streamlit
- Jupyter Notebook
