# Dataset:
Distribution: [1421, 1793, 2184, 6039, 9054]

# Premier modèle BERT :
encoder_url = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

Total params: 28764162 (109.73 MB)
Trainable params: 28764161 (109.73 MB)
Non-trainable params: 1 (1.00 Byte)

epochs = 5

## Entrainement:
### Small BERT
loss: 0.2665 - binary_accuracy: 0.9029   => epoch = 5, learning_rate = 3e-5, DropOut = 0.2, batch_size = 32
loss: 0.5887 - binary_accuracy: 0.9029   => epoch = 10, learning_rate = 3e-5, DropOut = 0.2, batch_size = 32   => Overfitting
loss: 0.2280 - binary_accuracy: 0.9024   => epoch = 5, learning_rate = 1e-5, DropOut = 0.2, batch_size = 32
loss: 0.2733 - binary_accuracy: 0.9034   => epoch = 10, learning_rate = 1e-5, DropOut = 0.2, batch_size = 32
loss: 0.2473 - binary_accuracy: 0.9029   => epoch = 5, learning_rate = 5e-6, DropOut = 0.4, batch_size = 16
loss: 0.3014 - binary_accuracy: 0.8683   => epoch = 5, learning_rate = 1e-6, DropOut = 0.2, batch_size = 32    => under-train

loss: 0.5034 - binary_accuracy: 0.7756   => gel de toutes les couches du BERT
### Base BERT
loss: 0.4425 - binary_accuracy: 0.9166   => epoch = 5, learning_rate = 3e-5, DropOut = 0.2, batch_size = 32
loss: 0.7411 - binary_accuracy: 0.9176   => epoch = 10, learning_rate = 3e-5, DropOut = 0.2, batch_size = 32    => Overfitting
loss: 0.2311 - binary_accuracy: 0.9176   => epoch = 5, learning_rate = 5e-6, DropOut = 0.2, batch_size = 32     => Best
loss: 0.2215 - binary_accuracy: 0.9137   => epoch = 5, learning_rate = 5e-6, DropOut = 0.2, batch_size = 32, warmupSteps * 2


Best:
Test Loss: 0.22647494077682495
Test Accuracy: 0.9136585593223572
Test Precision: 0.9454778432846069
Test Recall: 0.9412515759468079

## Axes d'amélioration
- Essayer d'entrainer un modèle BERT plus gros (large) avec un dataset plus gros
- Optimiser un peu plus en jouant sur le early stopping
















# Données de validation (val_ds) :
    Ce sont les données utilisées pendant l'entraînement pour évaluer la performance du modèle après chaque epoch.
    Ces résultats servent principalement à surveiller l'apprentissage, prévenir le sur-apprentissage (overfitting) et ajuster les hyperparamètres.
    Les résultats affichés ici (perte et précision) sont calculés sur les mêmes données que celles utilisées après chaque epoch dans le processus d'entraînement.
# Données de test (test_ds) :
    Ce sont des données complètement nouvelles et non vues par le modèle ni pendant l'entraînement ni pendant la validation.
    Ces données servent à évaluer de manière indépendante et finale la performance réelle du modèle sur des exemples non vus.
    Le résultat sur les données de test est un indicateur de la capacité du modèle à généraliser sur des données inconnues.


# Accuracy
L'accuracy est le pourcentage de prédictions correctes du modèle parmi l'ensemble des exemples.

# Loss
La loss mesure directement l'erreur entre les sorties brutes du modèle (logits avant activation, ou probabilités après activation) et les vérités terrain (labels).
Elle opère au niveau des probabilités ou des logits, ce qui signifie qu'elle prend en compte le niveau de confiance du modèle dans ses prédictions.
Pourquoi est-elle différente de l'accuracy ? :
L'accuracy se base sur les classes finales (après argmax ou un seuil appliqué sur la sortie softmax/sigmoïde).
La loss, elle, regarde les valeurs avant cette étape (par exemple, des probabilités ou logits). Ainsi, même si le modèle classe correctement un exemple (accuracy correcte), une prédiction avec une probabilité faible (incertitude élevée) augmentera la loss.

# F1-score
Le F1-score est une métrique qui pondère l'équilibre entre précision (faux positifs) et rappel (faux négatifs).
C'est utile pour les ensembles de données déséquilibrés où l'accuracy peut donner une impression trompeuse (par exemple, prédire toujours la classe majoritaire peut donner une bonne accuracy, mais un mauvais F1-score).
Ce que le F1-score fait : Évalue la qualité des prédictions positives et la capacité à capturer les cas pertinents.