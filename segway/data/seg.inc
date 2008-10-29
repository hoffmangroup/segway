#ifndef SEG_INC
#define SEG_INC

%% component types
#define COMPONENT_TYPE_DIAG_GAUSSIAN 0
#define COMPONENT_TYPE_GAMMA 3

%% cardinalities

% data exists or is missing?
#define CARD_NONMISSING 2

% which segment?
#define CARD_SEG 2

% AA .. TT
#define CARD_DINUCLEOTIDE 16

%% types

#define TYPE_SEG discrete hidden cardinality CARD_SEG

%% conditionalparents

#define CONDITIONALPARENTS_NONMISSING nil using DenseCPT("internal:UnityScore")

#endif
