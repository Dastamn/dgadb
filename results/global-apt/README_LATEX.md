# Intégration des Résultats dans LaTeX/Overleaf

## Guide rapide

### 1. Copier les fichiers

Dans votre projet Overleaf, créez un fichier `results.tex` et copiez le contenu de `latex_results.tex`.

### 2. Ajouter les packages

Dans le préambule de votre document LaTeX (avant `\begin{document}`), ajoutez les packages depuis `latex_preamble.tex` :

```latex
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{siunitx}
% ... etc (voir latex_preamble.tex)
```

### 3. Inclure la section

Dans votre document principal :

```latex
\input{results}  % ou \include{results}
```

## Structure de la section

La section résultats inclut :

1. **Description du dataset** - Caractéristiques et split
2. **Tableau de comparaison** - Table~\ref{tab:results} avec tous les algorithmes
3. **Findings clés** - Analyse des résultats
4. **Optimisation de seuil** - Explication de l'ajustement automatique
5. **Impact de la structure** - Pourquoi certains algorithmes peinent
6. **Figures** - Structure du réseau et pipeline

## Tableau des résultats

Le tableau utilise `booktabs` pour un style professionnel. Les colonnes incluent :
- ROC-AUC
- Precision
- Recall
- F1-Score
- True Positives (TP)
- False Positives (FP)

## Figures

### Option 1 : Convertir Mermaid en image

1. Exportez les diagrammes Mermaid depuis GitHub ou un outil en ligne
2. Sauvegardez en PDF ou PNG haute résolution
3. Utilisez `\includegraphics` dans LaTeX

### Option 2 : Utiliser TikZ

Un exemple TikZ basique est fourni dans `latex_results.tex`. Vous pouvez l'améliorer ou utiliser des outils comme :
- [Mermaid to TikZ converter](https://github.com/jfinkels/mermaid2tikz)
- Dessiner manuellement avec TikZ

### Option 3 : Utiliser des outils en ligne

- [Mermaid Live Editor](https://mermaid.live/) → Export SVG/PNG
- [Draw.io](https://app.diagrams.net/) pour créer des diagrammes

## Personnalisation

### Changer le style du tableau

```latex
% Pour un tableau plus compact
\begin{tabular}{l*{5}{c}}  % au lieu de lcccccc

% Pour ajouter des couleurs
\usepackage{xcolor}
\rowcolor{lightgray}  % pour une ligne
```

### Ajouter des statistiques

```latex
\subsection{Statistical Significance}

We performed paired t-tests comparing GraphSAGE to other methods...
% (ajoutez vos tests statistiques)
```

### Ajouter des graphiques

```latex
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[
    xlabel={Algorithm},
    ylabel={ROC-AUC},
    ybar,
    bar width=0.5cm,
]
\addplot coordinates {
    (GraphSAGE,0.97)
    (NetWalk,0.66)
    (SLADE,0.56)
    (GAT,0.52)
    (GeneralDYG,0.49)
};
\end{axis}
\end{tikzpicture}
\caption{ROC-AUC comparison across algorithms}
\label{fig:roc_comparison}
\end{figure}
```

## Exemple de citation dans le texte

```latex
As shown in Table~\ref{tab:results}, GraphSAGE achieved the best 
performance with a ROC-AUC of \num{0.970} and an F1-score of 
\num{0.862}. The network structure (Figure~\ref{fig:network_structure}) 
exhibits strong centralization, which explains the challenges 
faced by some algorithms.
```

## Checklist avant soumission

- [ ] Tous les packages nécessaires sont dans le préambule
- [ ] Les tableaux utilisent `booktabs` (style professionnel)
- [ ] Les figures sont en haute résolution (PDF ou PNG 300dpi+)
- [ ] Toutes les références (`\ref{}`) fonctionnent
- [ ] Les nombres sont formatés de manière cohérente
- [ ] Les unités utilisent `siunitx` si applicable
- [ ] Les algorithmes sont nommés de manière cohérente
- [ ] Les résultats correspondent aux fichiers JSON

## Conseils pour Overleaf

1. **Compilation** : Utilisez `pdflatex` ou `xelatex`
2. **Images** : Préférez PDF ou PNG haute résolution
3. **Tableaux longs** : Utilisez `longtable` si nécessaire
4. **Références** : Compilez 2-3 fois pour que les références se mettent à jour
5. **Collaboration** : Les fichiers `.tex` se synchronisent bien avec Git

## Exemple de structure de document complet

```latex
\documentclass[conference]{IEEEtran}  % ou article, etc.

% Preamble avec packages
\input{latex_preamble}

\begin{document}

\title{Anomaly Detection in Dynamic Graphs...}
\author{...}
\maketitle

\begin{abstract}
...
\end{abstract}

\section{Introduction}
...

\section{Related Work}
...

\section{Methodology}
...

\section{Experimental Setup}
...

\input{results}  % <-- Votre section résultats

\section{Discussion}
...

\section{Conclusion}
...

\bibliography{references}
\end{document}
```

## Support

Si vous avez des problèmes :
1. Vérifiez les logs de compilation dans Overleaf
2. Assurez-vous que tous les packages sont installés
3. Testez les figures une par une
4. Utilisez `\listoffigures` et `\listoftables` pour vérifier les références
