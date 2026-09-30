"""Un module par API externe. Chaque module :
- expose NAME (nom affiché) et ATTRIBUTION (mention exigée par la licence, ou None) ;
- lève euria.http.SourceError si la donnée est indisponible ;
- renvoie uniquement le format commun défini dans euria.schema."""
