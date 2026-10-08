# TRACE DES TRAVAUX — Extensions neuronales PyTorch pour MFGLib

**Statut : arrêt de développement — base de reprise**
**Date : 2026-10-08**
**Branche : `vibe/neural-mfg-algorithms-c210cd`**
**PR : https://github.com/slemenec/mfglib/pull/1 (draft)**

---

## 1. Objectif initial et périmètre couvert

Ajouter à MFGLib des algorithmes neuronaux basés PyTorch pour résoudre des
MFG finis, les benchmarquer sur les environnements MFGLib (Beach Bar en
particulier), les intégrer comme extensions locales de la librairie, et
documenter l'ensemble.

**Périmètre livré :**

| # | Livrable | Fichier |
|---|---|---|
| 1 | `NeuralFictitiousPlay` — logits tabulaires + Adam sur policy gradient analytique | `mfglib/alg/neural_fictitious_play.py` |
| 2 | `NeuralPolicyMLP` — MLP à poids partagés, fit KL vers BR dure + moyennage FP | `mfglib/alg/neural_policy_mlp.py` |
| 3 | `NeuralTwoNetworks` — 2 réseaux (politique + densité), apprentissage alterné, cohérence trackée | `mfglib/alg/neural_two_networks.py` |
| 4 | Exports `mfglib.alg` | `mfglib/alg/__init__.py` |
| 5 | Tests (paramétrés + dédiés cohérence) | `tests/test_algorithms.py` |
| 6 | Benchmark Beach Bar vs baselines | `benchmarks/benchmark_neural_fictitious_play.py` |
| 7 | Documentation LaTeX complète (théorie, schémas TikZ, comparaisons) | `docs/neural_fictitious_play.tex` |
| 8 | Présent fichier de trace | `docs/TRACE_TRAVAUX.md` |

---

## 2. Résultats de benchmark (Beach Bar, 300 itérations, défauts)

Exploitabilité finale (↓ mieux) :

| Algorithme | T=2, n=4 | T=5, n=10 | T=10, n=20 |
|---|---|---|---|
| **NeuralFictitiousPlay** (lr=1.0) | **1.1e-6** | 1.1e-1 | 4.7 |
| **NeuralPolicyMLP** | 5.8e-3 | 1.6e-2 | **5.7e-2** |
| **NeuralTwoNetworks** | 4.7e-2 | 2.2e-1 | 7.5e-1 |
| FictitiousPlay (baseline) | 2.0e-3 | 7.5e-3 | 3.2e-2 |
| OnlineMirrorDescent (α=10) | 1.8e-6 | 5.2e-1 | 4.7 |
| PriorDescent | 6.6e-1 | 4.2 | 4.3e1 |
| MFOMO | 2.1e-1 | 5.3 | 2.6e1 |

Faits saillants :
- `NeuralFictitiousPlay` égale OMD sur la petite instance (expl ~1e-6) et le
  bat sur toutes les autres à budget égal.
- `NeuralPolicyMLP` est le plus robuste à la taille : converge
  monotone sur les 3 instances ; à 2000 itérations sur n=20 il atteint
  9.1e-3 (43 s), proche du régime de FictitiousPlay.
- `NeuralTwoNetworks` fonctionne mais ~1 ordre de grandeur derrière
  NeuralPolicyMLP : plancher imposé par l'erreur du réseau densité.

## 3. Cohérence politique-densité (NeuralTwoNetworks)

Implémenté au dernier cycle (dernière demande utilisateur) :

- **Métrique** : Jeffreys (KL symétrisé) entre le champ moyen neural
  `L_ψ` et la forward exacte de `π_θ`, enregistrée à chaque itération dans
  `state.coherence_history` ; méthode publique `coherence(state)`.
  Empirique : décroît en tandem avec l'exploitabilité (4.9e-2 → 2.9e-2
  pendant que expl 0.80 → 0.47 sur T=5, n=10).
- **Pénalité `lambda_coh`** : ancre le fit politique sur la conditionnelle
  impliquée par le réseau densité `π_cond = L_ψ/Σ_a L_ψ` (terme
  `λ·KL(π_cond ‖ π_θ)` ajouté à la loss). Effet doux et dépendant de
  l'instance : améliore n=4 (4.7e-2→4.6e-2) et n=10 à λ=1 (2.5e-1→2.2e-1),
  neutre sur n=20. **Valeur surtout diagnostique** : l'historique de
  cohérence distingue un run limité par l'approximation de densité (Coh
  élevé) d'un run limité par le fit de politique (Coh bas, expl élevé).
- `lambda_coh` ajouté à l'espace de recherche Optuna (log-uniform [1e-3, 10]).

## 4. Décisions techniques clés (et pourquoi)

1. **PG pur rejeté pour le MLP** : policy gradient direct avec le MLP diverge
   sur Beach Bar (expl monte 0.3→2.3) — cible mobile. Diagnostic : le fit
   supervisé du même réseau vers la BR exacte atteint une KL de 4e-6, donc
   ce n'est pas un problème d'optimisation mais de boucle externe.
   Correctif : double boucle « fit interne vers BR + moyennage FP externe »
   (Heinrich & Silver 2016).
2. **BR dure plutôt que softmax(Q)** comme cible : la softmax à température
   fixe biaise la meilleure réponse ; le moyennage FP fournit déjà le
   lissage nécessaire. (La variante lissée reste disponible via `temp`.)
3. **`torch.manual_seed(0)` dans `init_state`** : runs reproductibles,
   objectif Optuna déterministe.
4. **Détachement systématique** des politiques retournées (`no_grad`,
   `detach`) : MFGLib a `filterwarnings = error`, et tout tenseur
   `requires_grad` qui atteint `exploitability_score` fait échouer les tests.
5. **Gradient analytique (`y.grad = grad`) plutôt qu'autograd** pour le
   tabulaire : gradient exact de la loss linéarisée, sans graphe inutile.
6. **Optuna fonctionnel sur les 3 algos**, y compris recherche
   d'architecture (hidden_dim, n_layers, time_encoding) pour les variantes
   MLP — testé manuellement : lr=0.546, tau=2.57 trouvés sur n=10, expl
   divisée par ~28.

## 5. Validation à date de l'arrêt

- `pytest tests/test_algorithms.py -k alg_on_lr` : **32 passed** (NE fermé
  Left-Right, 4 μ0, 8 algorithmes) + **2 tests cohérence** = 34.
- Suite non-regression (`not tuner and not alg_on_lr and not save_and_load`) :
  **108 passed**.
- `ruff`, `black`, `isort`, `mypy --strict` : **propres** sur tous les
  fichiers touchés (17 sources dans `mfglib/alg/`).
- Benchmark rejoué de bout en bout après le dernier changement.
- Doc LaTeX : équilibre d'environnements/labels/accolades vérifié
  programmatiquement (**pdflatex indisponible dans le sandbox — non compilé
  visuellement**).

## 6. Points d'arrêt précis / état exact du code

- `NeuralTwoNetworks.coherence()` et `lambda_coh` : **implémentés, testés
  (2 tests), benchmarkés, documentés (sec. coherence dans le .tex)** — mais
  les résultats cohérence (par. 3 ci-dessus) ne sont **pas encore dans les
  tableaux du .tex** (seul le texte de la section coherence les cite).
- Le commit de tête contient tout ce qui précède ; le PR #1 est **draft**,
  CI non encore observé (pas encore déclenché à date d'arrêt).

## 7. Pistes de reprise (par ordre de valeur estimée)

1. **Intégrer les chiffres cohérence au tableau comparatif du .tex** et
   ajouter une courbe cohérence vs exploitabilité (données déjà mesurées).
2. **`NeuralTwoNetworks` : découpler densité et fit politique** — mettre à
   jour la densité à *chaque* itération (au lieu d'alterner) coûte une
   forward pass par tour mais supprime l'instabilité de cible mobile ;
   comparer.
3. **Pénalité de cohérence sur la *densité*** (symétrique de celle
   implémentée) : ancre `L_ψ` sur la forward pendant le step politique.
4. **Rollouts échantillonnés** pour le réseau densité (remplacer la forward
   exhaustive) — motivation initiale de l'approche 2 réseaux, non encore
   implémentée ; c'est le gain réel attendu sur très grands |S|.
5. **Annealing de température** (`temp_n → 0`) pour NeuralPolicyMLP.
6. **Architecture équivariante** (CNN/tore) pour Beach Bar à très grand n.
7. **Tuner Optuna** : lancer `alg.tune()` complet sur les 3 instances pour
   figer des défauts par taille d'instance (les défauts actuels sont
   raisonnés, pas optimisés).
8. **Surveiller le CI du PR #1** et corriger si nécessaire.

## 8. Commandes utiles à la reprise

```bash
# Tests ciblés
python -m pytest tests/test_algorithms.py -k "alg_on_lr"      # ~4 min
python -m pytest tests/test_algorithms.py -k "coherence"     # ~5 s

# Benchmark complet (~5 min)
python benchmarks/benchmark_neural_fictitious_play.py

# Lint complet
ruff check mfglib/ tests/ benchmarks/
black --check mfglib/ tests/test_algorithms.py
mypy mfglib/alg/

# Utilisation type
from mfglib.alg import NeuralFictitiousPlay, NeuralPolicyMLP, NeuralTwoNetworks
from mfglib.env import Environment
alg = NeuralTwoNetworks(lambda_coh=1.0)
pis, expls, rts = alg.solve(Environment.beach_bar(), max_iter=300)
```

## 9. Historique des demandes utilisateur (trace décisionnelle)

1. Algorithme neural PyTorch pour MFG + benchmark beachbar + extension
   locale MFGLib → `NeuralFictitiousPlay` v1 (PG tabulaire).
2. Doc LaTeX implémentation + tests → `docs/neural_fictitious_play.tex`.
3. Question auto-tune / impact architecture → vérif Optuna OK, analyse
   tabulaire vs MLP.
4. « Pas de vrai réseau ? » → clarification tabulaire.
5. Vrai réseau pour grands espaces + bench + doc → `NeuralPolicyMLP`
   (après diagnostic de divergence du PG pur, double boucle).
6. « Policy gradient ? » → réponse : non pour le MLP (supervisé), oui pour
   le tabulaire ; détails.
7. Question 1 réseau 2 têtes → analyse faisabilité, contrainte de cohérence.
8. Variante 2 réseaux alternés → `NeuralTwoNetworks`.
9. « Livrer où ? » → branche + draft PR #1.
10. Documenter les 3 approches (avantages/inconvénients/schémas) → section
    comparative + TikZ dans le .tex.
11. Tracer la cohérence politique-densité → métrique + pénalité + tests.
12. **Présent fichier** : trace finale pour arrêt/reprise.
